import os
import json
import asyncio
import requests
from pathlib import Path
from dotenv import load_dotenv

# ====== FAIR-LLM IMPORTS ======
from fairlib import (
    HuggingFaceAdapter,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    SimpleAgent,
    SimpleReActPlanner,
    RoleDefinition,
)

# AbstractTool might be exposed at the top-level or in core.tools
try:
    from fairlib import AbstractTool
except ImportError:
    from fairlib.core.tools.abstract_tool import AbstractTool


# =========================================================
# 1. ENVIRONMENT & LLM SETUP (FAIR-LLM ADAPTER)
# =========================================================

load_dotenv()

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
OMDB_KEY = os.getenv("OMDB_KEY")
SERP_KEY = os.getenv("SERP_KEY")

if not HF_TOKEN:
    raise ValueError("HUGGING_FACE_HUB_TOKEN / HF_TOKEN not found in .env file.")
if not OMDB_KEY:
    raise ValueError("OMDB_KEY not found in .env file.")
if not SERP_KEY:
    raise ValueError("SERP_KEY not found in .env file.")

print("Loading language model via HuggingFaceAdapter...")
agent_llm = HuggingFaceAdapter(
    model_name="llama3-8b",
    auth_token=HF_TOKEN,
)
print("Model loaded! Ready to build the agent.")


# =========================================================
# 2. OMDB TOOL IMPLEMENTATION (AbstractTool)
# =========================================================

class OMDBMovieInfoTool(AbstractTool):
    """
    Tool that looks up movie information from the OMDB API.

    INPUT FORMAT (tool_input):
      - A JSON string with at least a "title" field:
        {"title": "Inception"}

      - Optional fields:
        {"title": "Dune", "year": "2021", "type": "movie"}
    """

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.name = "omdb_movie_info"
        self.description = (
            "Use this tool to look up detailed information about a movie or series "
            "from the OMDB API. Input MUST be a JSON string with at least a 'title' "
            "field, e.g. '{\"title\": \"Inception\"}'. Optionally include 'year' and 'type'. "
            "The tool returns a JSON object with fields like Title, Year, Genre, Plot, etc."
        )

    def _call_omdb(self, title: str, year: str | None = None, type_: str | None = None) -> dict:
        params = {
            "apikey": self.api_key,
            "t": title,
        }
        if year:
            params["y"] = year
        if type_:
            params["type"] = type_

        response = requests.get("http://www.omdbapi.com/", params=params, timeout=10)
        if response.status_code != 200:
            return {
                "Response": "False",
                "Error": f"HTTP error from OMDB: {response.status_code}",
            }

        try:
            data = response.json()
        except Exception as e:
            return {
                "Response": "False",
                "Error": f"Failed to parse OMDB JSON: {e}",
            }

        return data

    def use(self, tool_input: str, **kwargs) -> str:
        """
        FairLLM will pass a string `tool_input` here.
        We parse it as JSON, call OMDB, and return a JSON string result.
        """
        try:
            payload = json.loads(tool_input) if tool_input else {}
        except json.JSONDecodeError:
            # If the agent messed up, try to treat the input as the raw title
            payload = {"title": tool_input.strip()}

        title = (payload.get("title") or "").strip()
        year = (payload.get("year") or "").strip() or None
        type_ = (payload.get("type") or "").strip() or None

        if not title:
            return json.dumps(
                {
                    "Response": "False",
                    "Error": "No 'title' provided to omdb_movie_info tool.",
                }
            )

        data = self._call_omdb(title, year=year, type_=type_)

        # For the agent, keep it compact but structured
        if data.get("Response") == "True":
            result = {
                "Title": data.get("Title"),
                "Year": data.get("Year"),
                "Genre": data.get("Genre"),
                "Runtime": data.get("Runtime"),
                "imdbRating": data.get("imdbRating"),
                "Plot": data.get("Plot"),
                "Director": data.get("Director"),
                "Actors": data.get("Actors"),
                "Language": data.get("Language"),
                "Rated": data.get("Rated"),
            }
        else:
            result = {
                "Response": "False",
                "Error": data.get("Error", "Unknown OMDB error."),
            }

        return json.dumps(result)


class FavoriteMoviesTool(AbstractTool):
    """
    Reads a local 'favorites.txt' and returns the user's favorite movies.
    """

    def __init__(self, favorites_path: str | Path = "favorites.txt"):
        super().__init__()
        self.name = "favorite_movies"
        self.description = (
            "Reads a local 'favorites.txt' file and returns the user's favorite movies "
            "as JSON. The file should contain one movie title per line. "
            "IMPORTANT: Treat movies in this list as titles the user has ALREADY SEEN "
            "and likes. Use them ONLY to infer the user's taste (genres, directors, tone). "
            "Do NOT recommend these exact titles again unless the user explicitly asks "
            "for rewatch suggestions."
        )
        self.favorites_path = Path(favorites_path)

    def _read_favorites(self) -> list[str]:
        if not self.favorites_path.exists():
            return []

        favorites: list[str] = []
        with self.favorites_path.open("r", encoding="utf-8") as f:
            for line in f:
                title = line.strip()
                if not title:
                    continue
                if title.startswith("#"):
                    # allow comments
                    continue
                favorites.append(title)
        return favorites

    def use(self, tool_input: str, **kwargs) -> str:
        """
        We just read favorites.txt and return a JSON blob.
        """
        favorites = self._read_favorites()

        if not favorites:
            result = {
                "favorites": [],
                "count": 0,
                "file_path": str(self.favorites_path.resolve()),
                "message": (
                    "No favorites found. Ensure that favorites.txt exists and contains "
                    "one movie title per line."
                ),
            }
        else:
            result = {
                "favorites": favorites,
                "count": len(favorites),
                "file_path": str(self.favorites_path.resolve()),
                "message": f"Found {len(favorites)} favorite movies.",
            }

        return json.dumps(result)


# =========================================================
# 2b. BASE SERPAPI TOOL
# =========================================================

class BaseSerpApiTool(AbstractTool):
    """
    Base class for tools that query SerpAPI's /search endpoint.

    Subclasses should call _search(query, **extra_params) and then interpret
    the JSON result.
    """

    SERP_ENDPOINT = "https://serpapi.com/search"

    def __init__(self, api_key: str, engine: str = "google"):
        super().__init__()
        self.api_key = api_key
        self.engine = engine

    def _search(self, query: str, **extra_params) -> dict:
        """
        Perform a SerpAPI search against Google with a fixed location
        (Colorado Springs) to simulate a local user.

        Returns a dict:
          - {"status": "ok", "data": <parsed_json>}
          - or {"status": "error", "error": "...", "raw": "<text>"}
        """
        params = {
            "api_key": self.api_key,
            "engine": self.engine,
            "q": query,
            "location": "Colorado Springs, Colorado, United States",
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us",
        }
        params.update(extra_params)

        try:
            resp = requests.get(self.SERP_ENDPOINT, params=params, timeout=15)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Network error calling SerpAPI: {e}",
            }

        if resp.status_code != 200:
            return {
                "status": "error",
                "error": f"HTTP {resp.status_code} from SerpAPI",
                "raw": resp.text,
            }

        try:
            data = resp.json()
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse SerpAPI JSON: {e}",
                "raw": resp.text[:500],
            }

        return {"status": "ok", "data": data}


# =========================================================
# 2c. STREAMING AVAILABILITY TOOL
# =========================================================

class StreamingAvailabilityTool(BaseSerpApiTool):
    """
    Tool that uses SerpAPI to infer which streaming platforms carry a given movie.
    """

    KNOWN_SERVICES = [
        "Netflix",
        "Hulu",
        "Disney+",
        "Disney Plus",
        "Max",
        "HBO Max",
        "Amazon Prime Video",
        "Prime Video",
        "Apple TV+",
        "Peacock",
        "Paramount+",
        "Paramount Plus",
        "Starz",
        "Showtime",
        "Criterion Channel",
    ]

    def __init__(self, api_key: str):
        # normal google engine is fine here
        super().__init__(api_key=api_key, engine="google")
        self.name = "streaming_availability"
        self.description = (
            "Given a movie title (and optionally year), use SerpAPI to search "
            "for where it is available to stream (e.g., Netflix, Hulu, Max). "
            "Input MUST be a JSON string like '{\"title\": \"Dune\", \"year\": \"2021\"}'. "
            "Returns a JSON object listing likely streaming services."
        )

    @staticmethod
    def _parse_input(tool_input: str) -> dict:
        try:
            payload = json.loads(tool_input) if tool_input else {}
        except json.JSONDecodeError:
            payload = {"title": tool_input.strip()}
        return payload

    def use(self, tool_input: str, **kwargs) -> str:
        payload = self._parse_input(tool_input)
        title = (payload.get("title") or "").strip()
        year = (payload.get("year") or "").strip()

        if not title:
            return json.dumps({
                "title": None,
                "services": [],
                "message": "No 'title' provided to streaming_availability tool."
            })

        query_parts = [title]
        if year:
            query_parts.append(year)
        query_parts.append("where to watch streaming")
        query = " ".join(query_parts)

        search_result = self._search(
            query,
            # You can also use output/json_restrictor here if you want,
            # but the requirement was specifically for the theater tool.
            output="json",
            json_restrictor="organic_results",
        )
        if search_result.get("status") != "ok":
            return json.dumps({
                "title": title,
                "query": query,
                "services": [],
                "message": f"SerpAPI error: {search_result.get('error')}"
            })

        data = search_result["data"]
        organic = data.get("organic_results", []) or []

        found_services: set[str] = set()
        raw_top_results: list[dict] = []

        for res in organic[:10]:
            res_title = res.get("title", "") or ""
            snippet = res.get("snippet", "") or ""
            link = res.get("link", "") or ""

            raw_top_results.append({
                "title": res_title,
                "link": link,
            })

            text = f"{res_title} {snippet}".lower()
            for svc in self.KNOWN_SERVICES:
                if svc.lower() in text:
                    found_services.add(svc)

        services_list = sorted(found_services)
        if services_list:
            msg = f"Found streaming services: {', '.join(services_list)}"
        else:
            msg = "Could not confidently identify streaming services from search results."

        result = {
            "title": title,
            "query": query,
            "services": services_list,
            "raw_top_results": raw_top_results,
            "message": msg,
        }
        return json.dumps(result)


# =========================================================
# 2d. THEATRICAL SHOWTIMES TOOL WITH REQUIRED PARAMS
# =========================================================

class TheatricalShowtimesTool(BaseSerpApiTool):
    """
    Tool that uses SerpAPI (google_light engine) to get movies in theaters
    near Colorado Springs.

    REQUIRED PARAMS (for every actual SerpAPI call):
        params = {
            "engine": "google_light",
            "q": "Movies in theaters",
            "location": "Colorado Springs, Colorado, United States",
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us",
            "api_key": SERP_KEY,

            # Newly added:
            "output": "json",
            "json_restrictor": "organic_results"
        }

    Behaviors:
      - If called with empty input or an empty JSON object, it returns a list
        of all movies currently in theaters near Colorado Springs.
      - If called with a specific title, it filters that same result set to
        determine if/where the movie is playing.
    """

    THEATER_KEYWORDS = [
        "theatre", "theater", "cinema", "cinemas",
        "amc", "regal", "cine", "stadium", "imax"
    ]

    def __init__(self, api_key: str):
        # engine is still stored, but we *explicitly* build params ourselves
        super().__init__(api_key=api_key, engine="google_light")
        self.name = "theater_showtimes"
        self.description = (
            "Use this tool to query movies currently in theaters near Colorado Springs "
            "via SerpAPI google_light. Input options:\n"
            "  - Empty string or '{}' to get ALL movies currently in theaters.\n"
            "  - JSON string with a 'title' (and optional 'year') to check if that "
            "    movie is playing in local theaters.\n"
            "The underlying SerpAPI call ALWAYS uses:\n"
            "  engine='google_light', q='Movies in theaters', location='Colorado Springs, "
            "Colorado, United States', google_domain='google.com', hl='en', gl='us', "
            "output='json', json_restrictor='organic_results'."
        )

    @staticmethod
    def _parse_input(tool_input: str) -> dict:
        if not tool_input or tool_input.strip() == "":
            return {}
        try:
            payload = json.loads(tool_input)
            if isinstance(payload, dict):
                return payload
            return {}
        except json.JSONDecodeError:
            # Treat raw text as a title
            return {"title": tool_input.strip()}

    def _call_serp_movies_in_theaters(self) -> dict:
        """
        This is the ONLY place we hit SerpAPI for this tool.

        It uses exactly the required params (engine, q, location, google_domain,
        hl, gl, api_key, output, json_restrictor).
        """
        params = {
            "engine": "google_light",
            "q": "Movies in theaters",
            "location": "Colorado Springs, Colorado, United States",
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us",
            "api_key": self.api_key,
            "output": "json",
            "json_restrictor": "organic_results",
        }

        try:
            resp = requests.get(self.SERP_ENDPOINT, params=params, timeout=15)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Network error calling SerpAPI: {e}",
            }

        if resp.status_code != 200:
            return {
                "status": "error",
                "error": f"HTTP {resp.status_code} from SerpAPI",
                "raw": resp.text,
            }

        try:
            data = resp.json()
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to parse SerpAPI JSON: {e}",
                "raw": resp.text[:500],
            }

        return {"status": "ok", "data": data}

    def use(self, tool_input: str, **kwargs) -> str:
        """
        If no title is given: return all movies in theaters (based on organic_results).
        If a title is given: filter the same result set to infer whether it's in theaters.
        """
        payload = self._parse_input(tool_input)
        title = (payload.get("title") or "").strip()

        # Call SerpAPI once with the REQUIRED PARAMS.
        search_result = self._call_serp_movies_in_theaters()
        if search_result.get("status") != "ok":
            if not title:
                return json.dumps({
                    "mode": "all_in_theaters",
                    "title": None,
                    "query": "Movies in theaters",
                    "movies": [],
                    "message": f"SerpAPI error: {search_result.get('error')}",
                })
            else:
                return json.dumps({
                    "mode": "single_title",
                    "title": title,
                    "query": "Movies in theaters",
                    "in_theaters": False,
                    "theaters": [],
                    "message": f"SerpAPI error: {search_result.get('error')}",
                })

        data = search_result["data"]
        organic = data.get("organic_results", []) or []

        # Basic extraction of candidate "movies" from organic results.
        movie_candidates: list[str] = []
        raw_results: list[dict] = []

        for res in organic:
            res_title = (res.get("title") or "").strip()
            snippet = (res.get("snippet") or "").strip()
            link = (res.get("link") or "").strip()

            raw_results.append({
                "title": res_title,
                "snippet": snippet,
                "link": link,
            })

            # Very simple heuristic: take titles that look like movies
            # or showtime pages.
            if res_title:
                movie_candidates.append(res_title)

        # === CASE 1: ALL MOVIES IN THEATERS ===
        if not title:
            result = {
                "mode": "all_in_theaters",
                "title": None,
                "query": "Movies in theaters",
                "movies": movie_candidates,
                "raw_results": raw_results,
                "message": f"Found {len(movie_candidates)} organic results for 'Movies in theaters'."
            }
            return json.dumps(result)

        # === CASE 2: SPECIFIC TITLE (FILTER LOCALLY) ===
        title_lower = title.lower()
        theaters_for_title: list[dict] = []

        for r in raw_results:
            t = r["title"].lower()
            snippet = r["snippet"].lower()
            if title_lower in t or title_lower in snippet:
                # Only keep things that look theater-related
                if any(kw in t or kw in snippet for kw in self.THEATER_KEYWORDS):
                    theaters_for_title.append(r)

        in_theaters = len(theaters_for_title) > 0
        msg = (
            "Movie appears to be in theaters near Colorado Springs."
            if in_theaters
            else "No clear showtimes found near Colorado Springs."
        )

        result = {
            "mode": "single_title",
            "title": title,
            "query": "Movies in theaters",
            "in_theaters": in_theaters,
            "theaters": theaters_for_title,
            "raw_results": raw_results,
            "message": msg,
        }
        return json.dumps(result)


# =========================================================
# 3. TOOL REGISTRY, EXECUTOR, MEMORY, PLANNER, AGENT
# =========================================================

tool_registry = ToolRegistry()

omdb_tool = OMDBMovieInfoTool(api_key=OMDB_KEY)
tool_registry.register_tool(omdb_tool)

favorites_tool = FavoriteMoviesTool(favorites_path="favorites.txt")
tool_registry.register_tool(favorites_tool)

streaming_tool = StreamingAvailabilityTool(api_key=SERP_KEY)
tool_registry.register_tool(streaming_tool)

showtimes_tool = TheatricalShowtimesTool(api_key=SERP_KEY)
tool_registry.register_tool(showtimes_tool)

# Tool Executor
executor = ToolExecutor(tool_registry)

# Working Memory
memory = WorkingMemory()

# Planner
planner = SimpleReActPlanner(agent_llm, tool_registry)

planner.prompt_builder.role_definition = RoleDefinition(
    "You are a movie recommendation assistant.\n\n"
    "Your job on EACH user message is to:\n"
    "  • Use tools in a fixed order.\n"
    "  • Pick ONE main movie to recommend.\n"
    "  • Never recommend a movie the user has already seen.\n\n"
    "STRICT TOOL ORDER (EVERY QUERY):\n"
    "1) First, call 'theater_showtimes' ONCE with EMPTY INPUT (\"\" or \"{}\").\n"
    "   - This returns all movies currently in theaters near Colorado Springs.\n"
    "2) Second, call 'favorite_movies' ONCE.\n"
    "   - This returns the titles the user has ALREADY SEEN.\n"
    "3) Third (optional), call 'streaming_availability' for the ONE movie you are\n"
    "   most likely to recommend, if the user cares about streaming.\n"
    "4) Last, call 'omdb_movie_info' EXACTLY ONCE for your FINAL recommended movie.\n"
    "   - Do NOT call 'omdb_movie_info' more than once per user message.\n\n"
    "NEVER RECOMMEND FAVORITES:\n"
    "5) Treat every title returned by 'favorite_movies' as ALREADY SEEN.\n"
    "6) Do NOT recommend any of those titles unless the user explicitly asks for\n"
    "   rewatch suggestions.\n"
    "7) Before giving your final answer, quickly check your recommendation list and\n"
    "   remove any movie that appears in the favorites list. If that happens, pick\n"
    "   a different movie.\n\n"
    "THEATER VS STREAMING LOGIC:\n"
    "8) If the user says or implies they want to go to a theater (e.g., 'in theaters',\n"
    "   'cinema', 'showtimes', 'AMC', 'Regal', 'IMAX'), choose a movie from the\n"
    "   current in-theater list.\n"
    "9) Otherwise, you may choose either a current theatrical movie or a streaming\n"
    "   movie, whichever best matches their tastes.\n\n"
    "FINAL ANSWER FORMAT:\n"
    "10) Reply in plain natural language only.\n"
    "    - Do NOT show tool names, JSON, or internal reasoning.\n"
    "    - Clearly state your ONE main recommendation, why it fits their taste, and\n"
    "      where they can watch it (theater and/or streaming).\n"
)

# Agent Assembly
agent = SimpleAgent(
    llm=agent_llm,
    planner=planner,
    tool_executor=executor,
    memory=memory,
    max_steps=8,
)

print("Movie recommendation agent created with OMDB, favorites, streaming, and showtimes tools.")


# =========================================================
# 4. SIMPLE CLI LOOP TO INTERACT WITH THE AGENT
# =========================================================

async def chat_with_agent():
    print("\n🎬 Movie Recommender Agent (FairLLM) 🎬")
    print("Type something like: 'I liked Interstellar and Arrival, recommend similar movies.'")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print("Agent thinking...\n")
        try:
            response = await agent.arun(user_input)
        except Exception as e:
            print(f"[ERROR] Agent failed: {e}")
            continue

        print(f"Agent:\n{response}\n")


if __name__ == "__main__":
    asyncio.run(chat_with_agent())

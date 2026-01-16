# CareerMind AI - Copilot Instructions

## Architecture Overview

CareerMind AI is a **LangGraph-based multi-agent system** for intelligent career assistance. The architecture follows a **supervisor-worker pattern** where a routing supervisor dispatches user requests to specialized agents.

### Core Data Flow
1. User input enters via Streamlit UI (`app.py`)
2. Supervisor agent (`supervisor_node`) analyzes intent and routes to appropriate worker
3. Worker agents execute specialized tasks using domain-specific tools
4. Results aggregated and returned to UI with streaming callbacks
5. Chat history persisted via `StreamlitChatMessageHistory`

### Agent Tier System
- **Coordinator**: Supervisor (routes requests, manages context)
- **Specialists**: ResumeAnalyzer, JobSearcher, CoverLetterGenerator, WebResearcher
- **Expert Advisors**: CareerAdvisor, MarketAnalyst
- **Support**: ChatBot (fallback conversational agent)

## Critical Patterns & Conventions

### 1. **LangGraph State Management** (`agents.py`, `AgentState` TypedDict)
- All agents operate on shared `AgentState` containing: `user_input`, `messages`, `next_step`, `config`, `resume_data`, `job_preferences`, `analysis_results`, `error_count`
- State flows through graph nodes; each node validates and updates relevant fields
- **Strict typing required**: Use TypedDict for state, Pydantic for API contracts

### 2. **Supervisor Routing Logic** (`chains.py`, `get_supervisor_chain()`)
- Supervisor uses **intent-based keyword matching** with explicit rules per agent
- Resume queries → ResumeAnalyzer; Job mentions → JobSearcher; etc.
- **Routing must be deterministic**: same user input always reaches same agent
- Options come from `get_enhanced_team_members_details()` in `members.py`
- Fallback to ChatBot if no clear intent detected

### 3. **Tool Architecture** (`tools.py`)
- Tools are LangChain `StructuredTool` or `@tool` decorated functions
- Each agent has dedicated tool set (e.g., JobSearcher has `get_job_search_tool`, MarketAnalyst has `salary_analyzer_tool`)
- Tools return structured data (Pydantic models from `schemas.py`)
- **Error handling**: Return empty results rather than exceptions to prevent agent failures

### 4. **Prompt Engineering** (`prompts.py`)
- All agent prompts use `ChatPromptTemplate.from_messages()` pattern
- System prompt → Instructions → MessagesPlaceholder → agent_scratchpad
- **Domain-specific instructions** vary per agent (e.g., ResumeAnalyzer focuses on document parsing, not job search)
- Prompts reference agent capabilities from `members.py` profiles

### 5. **LLM Configuration** (`llms.py`, `EnhancedLLMManager`)
- Supports multiple providers: OpenAI (default), Groq, Anthropic
- Model selection via `get_best_model_for_task()` based on task type and tier
- **Cost-aware routing**: Fast/Budget tiers for simple tasks, Premium for complex reasoning
- Models configured with context window, max_tokens, streaming support

### 6. **Document Processing** (`data_loader.py`, `EnhancedResumeLoader`)
- Resume loading supports PDF, DOCX, TXT via PyMuPDF, python-docx
- Extracts text, metadata, and structure (sections, dates, skills)
- Cover letters generated as DOCX via `write_cover_letter_to_doc()`
- **Caching**: Processed resumes cached to avoid re-parsing

### 7. **Data Validation** (`schemas.py`)
- All inputs/outputs validated via Pydantic models (e.g., `JobSearchInput`, `JobResult`, `RouteSchema`)
- Enums for controlled values: `AgentType`, `EmploymentType`, `JobType`, `ExperienceLevel`
- Routes validated through `RouteSchema` with Literal field for agent names

### 8. **Search Integration** (`search.py`, `utils.py`)
- **Serper API** for Google search (job listings, company info)
- **FireCrawl API** for web scraping and content extraction
- **LinkedIn Job Scraper** for job fetching (synthetic data in fallback)
- Async throttling via `asyncio-throttle` to manage API rate limits

## Setup & Development

### Prerequisites
- Python 3.12+
- Environment file: `.streamlit/secrets.toml` with API keys:
  ```toml
  OPENAI_API_KEY = "sk-..."
  GROQ_API_KEY = "gsk_..."
  SERPER_API_KEY = "..."
  FIRECRAWL_API_KEY = "..."
  LANGCHAIN_API_KEY = "..."  # Optional: for LangSmith tracing
  ```

### Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Running Tests
No formal test suite exists; validation via:
- Unit imports: `python -c "import agents; import chains"`
- Streamlit checks: `streamlit run app.py --logger.level=debug`
- Agent routing: Test supervisor via `chains.get_supervisor_chain(llm)` with sample inputs

### LangSmith Tracing (Optional)
- Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in secrets
- Project name: `CareerMind-AI` (configurable in `app.py`)
- Trace calls to debug agent decisions and tool execution

## Common Development Tasks

### Adding a New Agent
1. Define agent profile in `members.py` with capabilities, tools, use_cases
2. Create agent-specific tools in `tools.py` (functions decorated with `@tool`)
3. Add system prompt in `prompts.py` with `get_<agent_name>_agent_prompt_template()`
4. Create agent executor in `agents.py` with `create_agent()` function
5. Add routing logic in `chains.py` and update `RouteSchema` with new agent name
6. Add agent node in `agents.py` graph definition

### Modifying Agent Behavior
- **Routing**: Edit supervisor keywords in `chains.py` (search for `CRITICAL ROUTING RULES`)
- **Tool outputs**: Adjust Pydantic models in `schemas.py`; agent tools automatically adapt
- **Prompts**: Modify system prompts in `prompts.py`; use f-strings for dynamic context
- **Max iterations**: Set in `create_agent()` `AgentExecutor` config (default: 3)

### Debugging Agent Decisions
- Enable verbose logging: `logging.basicConfig(level=logging.DEBUG)` at top of file
- Use LangSmith trace viewer for detailed execution graph
- Print state in node functions: `logger.info(f"State: {state}")`
- Check supervisor routing: Add print in `supervisor_node()` before returning next_step

## Key Integration Points

### External APIs
- **OpenAI/Groq**: LLM inference (agents and chains)
- **Serper**: Google search for jobs, companies, market data
- **FireCrawl**: Web scraping for job descriptions and company research
- **LinkedIn**: Job scraping (fallback to synthetic data)

### Streamlit Integration
- `StreamlitChatMessageHistory`: Persists messages per session
- `CustomStreamlitCallbackHandler`: Streams agent thoughts/tool calls to UI
- `st.session_state`: Stores resume data, preferences, session ID

## Performance Considerations

- **Agent max_iterations=3**: Prevents infinite loops; increase for complex reasoning
- **Tool result truncation**: Large documents summarized before agent processing
- **Async operations**: Job fetching and web research run asynchronously to improve responsiveness
- **Streaming**: LLM responses streamed to UI for perceived latency reduction
- **Error recovery**: Failed tool calls logged but don't crash agents (graceful degradation)

## File Mapping

| File | Purpose |
|------|---------|
| `agents.py` | LangGraph node functions, AgentState, agent executors |
| `app.py` | Streamlit UI, session management, callbacks |
| `chains.py` | Supervisor routing, chain definitions |
| `members.py` | Agent profiles, capabilities, team hierarchy |
| `tools.py` | LangChain tools for all agents (1600+ LOC) |
| `prompts.py` | System prompts per agent (1000+ LOC) |
| `schemas.py` | Pydantic models, enums, validation |
| `llms.py` | LLM provider configuration, model selection |
| `data_loader.py` | Resume/document parsing, cover letter generation |
| `search.py` | Job search, LinkedIn scraping |
| `utils.py` | Serper/FireCrawl clients, helpers |
| `custom_callback_handler.py` | Streamlit UI callbacks for streaming |

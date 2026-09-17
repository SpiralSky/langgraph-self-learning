# Self Learning
This is a simple **LangGraph Graph** meant to facilitate learning. Originally, it had much more scope, but was reduced,
and now mainly fulfills the simple purposes of formatting output prompts and making sure the user is not over-reliant on the AI model.

The composition of multiple models should also improve AI output for learning.

---

Note: Prompt results may vary.

---

### Running
To run the backend, simply do:
```commandline
cd backend
uv sync
uv run langgraph dev
```

To set up the frontend, simply change directories into agent-chat-ui and run:
```commandline
cd agent-chat-ui
npm install
npm run dev
```
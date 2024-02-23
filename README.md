# Chat with PDF using Mistral + Streamlit

## Setup

Get an API key from [mistral.ai](https://mistral.ai)

Create a file `.streamlit/secrets.toml` with a

```toml
MISTRAL_API_KEY="<YOUR_MISTRAL_KEY>"
```

Install uv ([what is uv?](https://github.com/astral-sh/uv))

```bash
pip install uv
```

Create virtual environment

```bash
uv venv # Create a virtual environment at .venv.
```

Activate virtual environment

```bash
# On macOS and Linux.
source .venv/bin/activate
```

install requirements

```bash
uv pip install -r requirements.txt
```

Run the app

```bash
streamlit run app.py
```

## improvements

- only the last message is sent to LLM, pass the last few messages to `client.chat_stream(...)`
- cache queries
- improve chunk retrieval for longer context questions
- sanitize inputs and outputs

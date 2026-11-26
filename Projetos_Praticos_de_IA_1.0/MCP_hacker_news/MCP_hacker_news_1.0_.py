import gradio as gr
import requests
from groq import Groq, GroqError


# Configuração da API Groq
api_key = "gsk_"

client = Groq(api_key=api_key)
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"


def resumo_noticias(
    tema: str,
    n: int = 5,
    modelo: str = "openai/gpt-oss-120b",
) -> str:
    """Busca manchetes do Hacker News e gera um resumo em português brasileiro.

    Args:
        tema: Assunto a pesquisar no Hacker News.
        n: Quantidade de manchetes a considerar, entre 3 e 10.
        modelo: Identificador do modelo disponível na API Groq.

    Returns:
        Um resumo de um parágrafo baseado somente nas manchetes encontradas.
    """
    tema = tema.strip()
    if not tema:
        return "Informe um tema para pesquisar."
    n = max(3, min(int(n), 10))

    try:
        response = requests.get(
            HN_SEARCH_URL,
            params={"query": tema, "tags": "story", "hitsPerPage": n},
            timeout=20,
        )
        response.raise_for_status()
        hits = response.json().get("hits", [])
    except (requests.RequestException, ValueError) as exc:
        return f"Erro ao consultar Hacker News: {type(exc).__name__}: {exc}"

    titulos = [
        titulo
        for hit in hits
        if (titulo := (hit.get("title") or "").strip())
    ]
    if not titulos:
        return f"Nenhuma manchete encontrada sobre '{tema}'."

    lista = "\n".join(f"- {titulo}" for titulo in titulos)
    prompt = (
        f"Tema: {tema}\n\nManchetes do Hacker News:\n{lista}\n\n"
        "Escreva um resumo curto em um único parágrafo, em português brasileiro. "
        "Use apenas as informações contidas nas manchetes. "
        "Não invente detalhes nem afirme ter lido os artigos completos."
    )

    try:
        completion = client.chat.completions.create(
            model=modelo,
            messages=[
                {
                    "role": "system",
                    "content": "Você resume manchetes de tecnologia com precisão e concisão.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
    except GroqError as exc:
        return f"Erro na API Groq: {type(exc).__name__}: {exc}"

    return (completion.choices[0].message.content or "").strip() or (
        "A Groq não retornou conteúdo."
    )


demo = gr.Interface(
    fn=resumo_noticias,
    inputs=[
        gr.Textbox(label="Tema", placeholder="Ex.: AI, Rust, Serverless"),
        gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Quantidade de manchetes"),
        gr.Dropdown(
            choices=[
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
            ],
            value="openai/gpt-oss-120b",
            label="Modelo Groq",
        ),
    ],
    outputs=gr.Textbox(label="Resumo", lines=6),
    title="Resumo de Notícias do Hacker News — Groq + MCP",
    description="Pesquise manchetes no Hacker News e receba um resumo gerado pela Groq.",
    api_name="resumo_noticias",
)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        mcp_server=True,
        show_error=True,
        inbrowser=True,
    )

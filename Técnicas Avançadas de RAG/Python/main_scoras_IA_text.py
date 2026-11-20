
"""
Introdução ao Nano‑GraphRAG
O Nano‑GraphRAG é uma implementação avançada de Graph Retrieval‑Augmented Generation (GraphRAG) que vai além da simples recuperação de trechos de texto. Ele integra técnicas de recuperação vetorial, construção e análise de grafos de conhecimento, além de incorporar uma camada de geração de respostas sofisticadas por meio de modelos de linguagem (LLMs). Essa abordagem híbrida permite que o sistema forneça respostas que combinam informações locais (como os chunks recuperados) com o contexto global extraído das relações entre entidades.

Principais Componentes e Funcionalidades
1. Divisão em Chunks
Objetivo e Funcionamento:
O Nano‑GraphRAG divide o documento de entrada em segmentos menores (chunks) para facilitar o processamento. Essa divisão é feita com base em um número fixo de palavras (por exemplo, 80 palavras) e utiliza sobreposição (por exemplo, 10 a 25 palavras) entre os chunks. Essa sobreposição é fundamental para manter o contexto entre os segmentos, evitando que informações importantes sejam perdidas nas fronteiras.

Importância:
Ao fragmentar o documento, o sistema possibilita uma indexação mais eficiente e uma recuperação precisa, tanto na abordagem local (via recuperação vetorial) quanto na construção da estrutura relacional.

2. Extração de Entidades e Relações
Método de Extração:
Para identificar entidades, o Nano‑GraphRAG utiliza uma heurística simples baseada em expressões regulares, que captura palavras que começam com letra maiúscula – frequentemente indicadoras de nomes próprios (como pessoas, organizações e locais).
Em seguida, ele estabelece relações do tipo “RELACIONADO_A” entre cada par consecutivo de entidades extraídas de cada chunk.

Integração com o Grafo:
Diferente de abordagens mais simples, onde as relações são apenas acumuladas em listas, o Nano‑GraphRAG atualiza dinamicamente um grafo representado em memória (por exemplo, utilizando um dicionário do tipo defaultdict(list)). Essa estrutura permite a modelagem das interconexões entre entidades, fornecendo uma base para análises globais do documento.

3. Construção e Exploração do Grafo
Construção do Grafo de Conhecimento:
O grafo é montado a partir das entidades e relações extraídas. Embora implementações anteriores utilizassem bibliotecas como o NetworkX para construir o grafo em memória, o Nano‑GraphRAG adota uma abordagem mais flexível, mantendo o grafo em um dicionário que pode ser atualizado dinamicamente.

Busca de Conexões Indiretas:
Uma funcionalidade destacada é a capacidade de realizar buscas de caminhos entre entidades. Por meio de uma busca em profundidade (DFS) com limite de profundidade (por exemplo, 3 níveis), o sistema identifica conexões indiretas entre entidades. Essa funcionalidade enriquece o contexto, permitindo que o sistema capture relações não óbvias que podem ser essenciais para a compreensão global do documento.

4. Vetorização e Recuperação com FAISS
Geração de Embeddings:
Utilizando o modelo SentenceTransformer (por exemplo, 'paraphrase-multilingual-MiniLM-L12-v2'), o Nano‑GraphRAG converte cada chunk em um vetor que representa seu significado semântico.

Indexação com FAISS:
Os embeddings gerados são indexados usando a biblioteca FAISS, que possibilita buscas por similaridade de forma rápida e eficiente. Essa etapa permite recuperar os trechos de texto mais relevantes com base na distância L2 entre os vetores.

Integração com a Análise do Grafo:
A recuperação vetorial fornece uma abordagem “local”, que, combinada com a análise “global” do grafo (relações e caminhos), cria uma base robusta para a geração de respostas.

5. Integração com LLM para Geração de Resposta
Construção do Prompt:
Uma das inovações do Nano‑GraphRAG é a integração com modelos de linguagem para a geração de respostas. A função responsável constrói um prompt que combina informações dos trechos recuperados via FAISS (info vetorial) com os dados estruturados do grafo (relações diretas e caminhos).
Chamada à API do LLM:
Este prompt é enviado a um LLM – por exemplo, ao modelo “gpt-4o-mini” – que gera uma resposta final. O LLM é orientado a integrar e sintetizar as informações dos dois domínios, produzindo uma resposta contextualizada que une detalhes locais com a visão global do conhecimento.
6. Componentes de Armazenamento e Processamento
Persistência e Armazenamento:
O Nano‑GraphRAG integra componentes de armazenamento, como KV Storage, NanoVectorDBStorage para os embeddings, e Graph Storage (por exemplo, utilizando NetworkX ou integração com Neo4J) para manter o estado do grafo e possibilitar atualizações incrementais.

Processamento Assíncrono:
A implementação faz uso intensivo de programação assíncrona, permitindo que múltiplas operações (como chamadas a APIs, extração, indexação e armazenamento) ocorram simultaneamente. Essa abordagem melhora a escalabilidade e a robustez do sistema em ambientes de produção.

Modularidade e Tipagem:
Utilizando bibliotecas como pydantic, dataclasses e dspy, o Nano‑GraphRAG é altamente modular e fortemente tipado, facilitando a manutenção, validação e extensão do código.

7. Suporte a Múltiplos Provedores e Refinamento Iterativo
Integração com Diferentes LLMs:
O sistema permite a configuração para utilizar diferentes provedores de LLM, como OpenAI, Azure OpenAI ou Amazon Bedrock, oferecendo flexibilidade conforme requisitos de custo e desempenho.

Refinamento Iterativo:
Por meio de módulos como o TypedEntityRelationshipExtractor e funções de crítica e refinamento (CritiqueCombinedExtraction e RefineCombinedExtraction), o Nano‑GraphRAG pode iterar sobre a extração de entidades e relações, aprimorando a qualidade dos dados e, consequentemente, a precisão das respostas.

8. Comparação com Implementações Anteriores
GraphRAG Simples:
Utiliza métodos básicos de divisão em chunks, extração de entidades e construção de grafos em memória (com NetworkX).
Foca na demonstração dos conceitos fundamentais sem integração avançada com LLM ou refinamento iterativo.
GraphRAG com Neo4J:
Incorpora um banco de dados Neo4J para armazenar e consultar o grafo, permitindo consultas mais sofisticadas por meio de Cypher e oferecendo persistência.
Destaca-se pela robustez e escalabilidade, mas o foco está na estruturação relacional e consulta dinâmica.
Nano‑GraphRAG (com LLM):
Combina a recuperação vetorial (via FAISS) com uma análise relacional rica (grafo em memória, busca de caminhos) e integra a geração de respostas por meio de LLMs.
Oferece refinamento iterativo, suporte a múltiplos provedores, processamento assíncrono e componentes de armazenamento persistente, resultando em um sistema robusto e flexível para aplicações de produção."""


from nano_graphrag import GraphRAG, QueryParam
import asyncio
import sys
import shutil
import os

import os



from nano_graphrag import GraphRAG, QueryParam

class RAGInterface:
    def __init__(self, working_dir="./dickens"):
        self.graph_func = GraphRAG(
            working_dir=working_dir,
            enable_naive_rag=True
        )
        self.text_loaded = False

    async def load_text(self, file_path="./ia.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        await self.graph_func.ainsert(text)
        self.text_loaded = True

        print("Texto carregado com sucesso!")
    
    
    async def query_text(self, question, mode):
        if not self.text_loaded:
            print("\nErro: Por favor, carregue o texto primeiro!")
            return
        
        try:
            param = QueryParam(mode=mode)
            response = await self.graph_func.aquery(question, param=param)
            return response
        except Exception as e:
            print(f"\nErro durante a consulta: {str(e)}")
            return "Desculpe, ocorreu um erro ao processar sua pergunta."

async def main():
    rag = RAGInterface()
    
    # Carrega o texto
    print("\nCarregando o texto...")
    await rag.load_text(
        r"C:\Users\eugen\Documents\.venv-cuda312\ia.txt"
    )

    while True:
        print("\nEscolha o modo do RAG:")
        print("1. GraphRAG (Local mode)")
        print("2. GraphRAG (Global mode)")
        print("3. Naive RAG")
        print("4. Sair")
        
        choice = input("\nEscolha uma opção (1-4): ").strip()
        
        if choice == "4":
            print("\nTchau, até breve!")
            break
            
        if choice not in ["1", "2", "3"]:
            print("\nInvalid choice! Please select 1, 2, or 3.")
            continue
            
        mode = {
            "1": "local",
            "2": "global",
            "3": "naive"
        }[choice]
        
        while True:
            question = input("\nDigite sua pergunta (ou 'voltar' para mudar o modo, 'sair' para sair): ").strip()
            
            if question.lower() == 'voltar':
                break
            elif question.lower() == 'sair':
                print("\nTchau, até breve!")
                sys.exit(0)
            elif question:
                print("\nProcessando sua pergunta...")
                response = await rag.query_text(question, mode)
                print(f"\nResposta: {response}")
            else:
                print("\nPor favor, digite uma pergunta válida!")

if __name__ == "__main__":
    asyncio.run(main())
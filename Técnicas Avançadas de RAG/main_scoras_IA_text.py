


from nano_graphrag import GraphRAG, QueryParam
import asyncio
import sys
import shutil
import os

import os

os.environ["OPENAI_API_KEY"] = "sk-proj"
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
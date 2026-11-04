


import os
import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_graph import End

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

def criar_agente():
    """Cria e retorna uma instância de agente configurada"""
    return Agent(
        'groq:openai/gpt-oss-120b', 
        system_prompt="Seja preciso e direto nas respostas. Forneça informações factuais."
    )

async def iterar_fluxo(agente):
    """
    Itera manualmente sobre o grafo de execução do agente.
    
    Esta função demonstra como ter acesso aos nós individuais do fluxo,
    permitindo inspeção detalhada do processamento do agente.
    """
    print("\n--- Iniciando iteração sobre o grafo de execução ---")
    
    # Use async with em vez de with para contexto assíncrono
    async with agente.iter("Qual é a capital do Ceará?") as agent_run:
        print("Executando o fluxo e coletando nós...")
        
        # Lista para armazenar os nós visitados
        nodes = []
        
        # Iterar sobre cada nó do grafo à medida que são processados
        async for node in agent_run:
            nodes.append(node)
            node_type = type(node).__name__
            
            # Exibir informação sobre o nó atual
            print(f"  - Processando nó: {node_type}")
            
            # Se for um nó final, exibir mais informações
            if isinstance(node, End):
                print(f"  - Nó final encontrado.")
                
                # Vamos tentar acessar outros atributos disponíveis
                print(f"  - Atributos disponíveis: {dir(node)[:10]}...")
        
        # Após a iteração, mostrar o resultado
        print("\nIteração completa!")
        print(f"Total de nós processados: {len(nodes)}")
        print(f"Tipos de nós: {set(type(n).__name__ for n in nodes)}")
        print("\nResultado Final:", agent_run.result.output)
    
    return agent_run.result

def main():
    """Função principal que executa todo o fluxo"""
    # Verificar se temos a API key
    if not os.getenv("GROQ_API_KEY"):
        print("Erro: GROQ_API_KEY não encontrada no ambiente ou arquivo .env")
        print("Crie um arquivo .env com GROQ_API_KEY=sua_chave_api")
        return
    
    print("=== Demonstração de Iteração Manual Sobre o Grafo de Execução ===")
    
    # Criar agente
    agente = criar_agente()
    
    # Executar a iteração sobre o fluxo
    asyncio.run(iterar_fluxo(agente))
    
    print("\n=== Demonstração concluída! ===")

if __name__ == "__main__":
    main() 
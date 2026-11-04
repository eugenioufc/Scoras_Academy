#!/usr/bin/env python3
"""
Exemplo de execução do PydanticAI usando método síncrono e streaming.
Para executar: python teste_streaming.py
"""

import os
import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

def criar_agente():
    """Cria e retorna uma instância de agente configurada"""
    return Agent(
        'groq:openai/gpt-oss-120b', 
        system_prompt="Seja preciso e direto nas respostas."
    )

def executar_sincrono(agente, pergunta):
    """Executa uma consulta de forma síncrona"""
    print(f"\n--- Executando consulta síncrona: '{pergunta}' ---")
    result = agente.run_sync(pergunta)
    print(f"Resposta: {result.output}")
    return result

async def executar_streaming(agente, pergunta):
    """Executa uma consulta usando streaming"""
    print(f"\n--- Executando consulta via streaming: '{pergunta}' ---")
    print("Resposta:")
    
    async with agente.run_stream(pergunta) as stream:
        # Para ver a resposta em tempo real, você poderia usar este loop:
        # async for chunk in stream:
        #     print(chunk, end="", flush=True)
        
        # Ou para obter a resposta completa no final:
        resposta = await stream.get_output()
        print(resposta)
    
    return "Streaming concluído"

async def demonstracao_streaming():
    """Demonstra streaming de tokens em tempo real"""
    print("\n--- Demonstração de streaming em tempo real ---")
    agente = criar_agente()
    
    pergunta = "Explique o que é machine learning em quatro frases curtas."
    print(f"Pergunta: {pergunta}")
    print("Resposta: ", end="", flush=True)
    
    # Simplificar para usar o método que sabemos funcionar
    async with agente.run_stream(pergunta) as stream:
        # Aqui usamos o método get_data() que já vimos funcionando
        resposta = await stream.get_output()
        print(resposta)
    
    print("\nStreaming em tempo real concluído!")

async def executar_async():
    """Função assíncrona principal"""
    # Criar o agente
    agente = criar_agente()
    
    # Executar consultas
    await executar_streaming(agente, "Qual é a capital da França?")
    
    # Demonstração de streaming em tempo real
    await demonstracao_streaming()
    
    return "Todas as consultas foram concluídas!"

def main():
    """Função principal que executa todo o fluxo"""
    # Verificar se temos a API key
    if not os.getenv("GROQ_API_KEY"):
        print("Erro: GROQ_API_KEY não encontrada no ambiente ou arquivo .env")
        print("Crie um arquivo .env com GROQ_API_KEY=sua_chave_api")
        return
    
    print("=== Teste de PydanticAI com execução síncrona e streaming ===")
    
    # Criar agente
    agente = criar_agente()
    
    # Executar consulta síncrona
    executar_sincrono(agente, "Qual é a capital da Itália?")
    
    # Executar consultas assíncronas
    asyncio.run(executar_async())
    
    print("\n=== Teste concluído! ===")

if __name__ == "__main__":
    main()
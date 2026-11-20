#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
import os
from dotenv import load_dotenv
import openai
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from pathlib import Path
load_dotenv()

class GraphRAGAvancado:
    def __init__(self):
        print("Inicializando GraphRAG com LLM...")
        
        print("Carregando modelo de embeddings...")
        self.modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        #Converte texto em vetores numéricos (embeddings)
        #Captura o significado semântico das frases
        #Permite comparar similaridade entre texto
        
        print("Preparando banco vetorial FAISS...")
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        
        self.chunks = []
        self.relacoes = []
        self.id_to_chunk = {}
        self.grafo = defaultdict(list)  # Estrutura de grafo simplificada

    def dividir_documento(self, documento: str, tamanho: int = 80, 
                         sobreposicao: int = 10) -> List[str]:
        palavras = documento.split()
        chunks = []
        inicio = 0
        while inicio < len(palavras):
            fim = inicio + tamanho
            chunk = " ".join(palavras[inicio:fim])
            chunks.append(chunk)
            inicio += tamanho - sobreposicao
        return chunks

    def extrair_entidades(self, chunk: str) -> List[str]:
        return re.findall(r'\b[A-Z][a-zA-Z]+\b', chunk)

    def extrair_relacoes(self, chunk: str) -> List[Tuple[str, str, str]]:
        entidades = self.extrair_entidades(chunk)
        relacoes = []
        for i in range(len(entidades) - 1):
            relacoes.append((entidades[i], "RELACIONADO_A", entidades[i+1]))
            # Adiciona ao grafo
            self.grafo[entidades[i]].append(entidades[i+1])
            self.grafo[entidades[i+1]].append(entidades[i])
        return relacoes

    def encontrar_caminhos(self, inicio: str, fim: str, max_depth: int = 3) -> List[List[str]]:
        def dfs(atual: str, destino: str, caminho: List[str], profundidade: int) -> List[List[str]]:
            if profundidade > max_depth:
                return []
            if atual == destino:
                return [caminho]
            caminhos = []
            for vizinho in self.grafo[atual]:
                if vizinho not in caminho:
                    novos_caminhos = dfs(vizinho, destino, caminho + [vizinho], profundidade + 1)
                    caminhos.extend(novos_caminhos)
            return caminhos
        
        return dfs(inicio, fim, [inicio], 0)

    def construir_base_conhecimento(self, documento: str):
        print("Construindo base de conhecimento...")
        
        self.chunks = self.dividir_documento(documento)
        embeddings = self.modelo.encode(self.chunks)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.id_to_chunk = {i: chunk for i, chunk in enumerate(self.chunks)}
        
        for chunk in self.chunks:
            self.relacoes.extend(self.extrair_relacoes(chunk))
        
        print(f"Processados {len(self.chunks)} chunks e {len(self.relacoes)} relações")

    def buscar_chunks_similares(self, pergunta: str, top_k: int = 3) -> List[str]:
        emb_pergunta = self.modelo.encode([pergunta])
        D, I = self.index.search(np.array(emb_pergunta, dtype=np.float32), top_k)
        return [self.id_to_chunk[i] for i in I[0] if i != -1]

    def buscar_relacoes_relevantes(self, pergunta: str) -> List[Dict]:
        palavras = pergunta.lower().split()
        resultados = []
        entidades_pergunta = set(self.extrair_entidades(pergunta))
        
        # Busca direta
        for ent1, rel, ent2 in self.relacoes:
            if any(p in ent1.lower() or p in ent2.lower() for p in palavras):
                resultados.append({
                    "entidade1": ent1,
                    "relacao": rel,
                    "entidade2": ent2,
                    "tipo": "direto"
                })
        
        # Busca por caminhos
        chaves_grafo = list(self.grafo.keys())  # Cria uma cópia das chaves
        for ent1 in entidades_pergunta:
            for ent2 in chaves_grafo:  # Usa a cópia em vez do dicionário direto
                if ent2 not in entidades_pergunta:
                    caminhos = self.encontrar_caminhos(ent1, ent2)
                    for caminho in caminhos:
                        if len(caminho) > 1:
                            resultados.append({
                                "entidade1": caminho[0],
                                "relacao": "CONECTADO_A",
                                "entidade2": caminho[-1],
                                "tipo": "caminho",
                                "caminho": " -> ".join(caminho)
                            })
        
        return resultados[:20]

    def gerar_resposta_llm(self, pergunta: str, info_vetorial: str, 
                          info_grafo: List[Dict]) -> str:
        # Separa relações diretas e caminhos
        relacoes_diretas = [r for r in info_grafo if r["tipo"] == "direto"]
        caminhos = [r for r in info_grafo if r["tipo"] == "caminho"]
        
        # Formata o contexto estruturado
        contexto_grafo = "Relações diretas encontradas:\n"
        for r in relacoes_diretas:
            contexto_grafo += f"- {r['entidade1']} --[{r['relacao']}]--> {r['entidade2']}\n"
        
        if caminhos:
            contexto_grafo += "\nCaminhos relevantes encontrados:\n"
            for r in caminhos:
                contexto_grafo += f"- {r['caminho']}\n"

        prompt = f"""
        Você é um assistente especializado em análise de conhecimento estruturado e não estruturado.
        
        Analise cuidadosamente as duas fontes de informação:

        1. TRECHOS DE TEXTO RELEVANTES:
        {info_vetorial}

        2. ESTRUTURA DE CONHECIMENTO:
        {contexto_grafo}

        Para responder à pergunta, você deve:
        - Usar o contexto dos trechos de texto para informações detalhadas
        - Considerar as relações do grafo para entender conexões
        - Explicar como as entidades se relacionam quando relevante
        - Integrar ambas as fontes em uma resposta coerente
        
        PERGUNTA: "{pergunta}"
        
        Forneça uma resposta estruturada que:
        1. Introduza o tema
        2. Desenvolva a resposta usando ambas as fontes
        3. Conclua conectando os pontos principais
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Você é um assistente especialista que integra informações textuais e estruturais para fornecer respostas curtas e precisas com menos de 300 caracteres."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Erro ao gerar resposta: {str(e)}"

    
from pypdf import PdfReader

def carregar_pdfs(arquivos: list[str]) -> str:
    documentos = []

    for arquivo in arquivos:
        leitor = PdfReader(arquivo)

        texto = "\n".join(
            pagina.extract_text() or ""
            for pagina in leitor.pages
        )

        if texto.strip():
            documentos.append(f"Documento: {arquivo}\n{texto}")
            print(f"PDF carregado: {arquivo}")
        else:
            print(f"Aviso: nenhum texto extraído de {arquivo}")

    if not documentos:
        raise ValueError("Nenhum texto foi extraído dos PDFs.")

    return "\n\n".join(documentos)



    # [O mesmo texto extenso que você já está usando]
    

def main():
    desktop = Path.home() / "Desktop"

    arquivos = [
        desktop / "Elsevier.pdf",
        desktop / "paper_eugenio_MSc.pdf"
    ]

    for arquivo in arquivos:
        if not arquivo.is_file():
            raise FileNotFoundError(
                f"PDF não encontrado: {arquivo}"
            )

    documento = carregar_pdfs([
        str(arquivo) for arquivo in arquivos
    ])

    rag = GraphRAGAvancado()
    rag.construir_base_conhecimento(documento)

    print("\nSistema pronto! Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("\nSua pergunta: ")

        if pergunta.lower() == "sair":
            break

        chunks = rag.buscar_chunks_similares(pergunta)
        relacoes = rag.buscar_relacoes_relevantes(pergunta)

        info_vetorial = "\n".join(chunks)

        print("\nChunks relevantes:")
        print(info_vetorial)

        print("\nRelações encontradas:")

        for r in relacoes:
            if r["tipo"] == "direto":
                print(
                    f"{r['entidade1']} "
                    f"--[{r['relacao']}]--> "
                    f"{r['entidade2']}"
                )
            else:
                print(f"Caminho: {r['caminho']}")

        print("\nResposta elaborada:")
        print(
            rag.gerar_resposta_llm(
                pergunta,
                info_vetorial,
                relacoes
            )
        )


if __name__ == "__main__":
    main()


    
    
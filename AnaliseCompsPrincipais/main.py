import sys
import csv
import numpy as np
from statistics import mean
from statistics import stdev

def pegarDados():
	matriz = list()
	
	entrada = sys.argv[1]
	with open(entrada) as dados:
		reader = csv.reader(dados)
		next(reader)
		
		while True:
			try:
				matriz.append(next(reader))
			except StopIteration:
				break

	return np.array(matriz)

def normalizar(matriz):
	transposta = matriz.T
	medias = tuple(mean(variavel) for variavel in transposta)
	dps = tuple(stdev(variavel) for variavel in transposta)

	for i in range(matriz.shape[0]):
		for j in range(matriz.shape[1]):
			matriz[i, j] = (matriz[i, j] - medias[j])/dps[j]

def aplicarPCA(dados):
    # Normaliza os dados
	normalizar(dados)

    # Calculo da matriz de covariancia (numpy.cov espera que cada linha seja uma variável)
    matrizCorr = np.cov(dados.T)

    # Autovalores e respectivos autovetores da matriz de covariancia
    autovals, autovets = np.linalg.eig(matrizCorr)
    autoPares = list(zip(autovals, autovets))

    # Funçao para ordenar pares (autovalor, autovetor) por autovalores
    def ordemOrdenacao(par):
        return par[0]

    autoPares.sort(key = ordemOrdenacao, reverse = True)

    # Escolha de autovetores que expliquem ao menos 90% da variância
    traco = sum(autovals)
    varExplicada = [(ap[0]/traco)*100 for ap in autoPares]
    varExpCum = np.cumsum(varExplicada)

    n = 0
    for varAcumulada in varExpCum:
        n += 1
        if varAcumulada >= 0.9:
            break

    # Obtencao da matriz de projecao
    nVariaveis = len(dados[0])

    autoVetsEscolhidos = tuple(autoPares[i, 1].reshape(nVariaveis, 1) for i in range(n))
    matrizProj = hstack(autoVetsEscolhidos)

    # Mapeamento dos dados no espaço definido pelos autovetores
    resultado = np.dot(dados, matrizProj)
    return resultado


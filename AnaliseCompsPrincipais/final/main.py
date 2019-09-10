import sys
import csv
import numpy as np
import statistics as stats

def pegarDados(endereco):
    matriz = list()

    headers = []
    typ = [[]]
    with open(endereco) as dados:
        reader = csv.reader(dados)
        headers = np.array(next(reader))[1:]
        
        while True:
            try:
                proxLinha = next(reader)
                linha = [float(x) for x in proxLinha[1:-1]]
                matriz.append(linha)

                typ[0].append(int(proxLinha[-1]))
            except StopIteration:
                break

    typ = np.array(typ).transpose()
    return {"headers":headers, "body":np.array(matriz), "types":typ}

def normalizar(matriz):
	transposta = matriz.transpose()
	medias = tuple(stats.mean(variavel) for variavel in transposta)
	dps = tuple(stats.stdev(variavel) for variavel in transposta)

	for i in range(matriz.shape[0]):
		for j in range(matriz.shape[1]):
			matriz[i, j] = (matriz[i, j] - medias[j])/dps[j]

def aplicarPCA(dados):
    #Normaliza os dados
    normalizar(dados)
        
    #Calculo da matriz de covariancia
    matrizCov = np.cov(dados.transpose())

    #Autovalores e respectivos autovetores da matriz de covariancia
    autovals, autovets = np.linalg.eig(matrizCov)
    autoPares = list(zip(autovals, autovets))

    #Ordenacao decrescente dos autovalores
    def ordemOrdenacao(par):
        return par[0]

    autoPares.sort(key = ordemOrdenacao, reverse = True)

    #Escolha de autovetores que expliquem ao menos 90% da variância
    traco = sum(autovals)
    varExplicada = [(ap[0]/traco) for ap in autoPares]
    
    varExpAcum = np.cumsum(varExplicada)
    n = 0
    for varAcumulada in varExpAcum:
        n += 1
        if varAcumulada >= 0.90:
            break

    #Obtencao da matriz de projecao
    nVariaveis = len(dados[0])

    autoVetsEscolhidos = tuple(autoPares[i][1].reshape(nVariaveis, 1) for i in range(n))
    matrizProj = np.hstack(autoVetsEscolhidos)

    #Mapeamento dos dados no espaço definido pelos autovetores
    resultado = np.dot(dados, matrizProj)
    return resultado

def colocarSaida(endereco, dados):
    nColunas = len(dados[0])
    headers = [[]]
    headers[0].append("Type")
    for i in range(1, nColunas):
        headers[0].append("CP" + str(i))

    headers = np.array(headers)
    dados = np.vstack((headers, dados)).tolist()
    for linha in range(1, len(dados)):
        dados[linha][0] = int(float(dados[linha][0]))
        for coluna in range(1, nColunas):
            dados[linha][coluna] = "{0:.2f}".format(float(dados[linha][coluna]))

    with open(endereco, "w") as f:
        writer = csv.writer(f, delimiter=",")
        for linha in dados:
            writer.writerow(linha)

enderecoEntrada = sys.argv[1]
entrada = pegarDados(enderecoEntrada)

saida = aplicarPCA(entrada["body"])
saida = np.hstack((entrada["types"], saida))

enderecoSaida = sys.argv[2]
colocarSaida(enderecoSaida, saida)
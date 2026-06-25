import os
import numpy as np
from scipy.io import wavfile
from scipy import signal

PASTA_AUDIOS = "C:/Users/miria/Documents/UFAL/TCC/codigos/arquivos/A2_9V/Anormal"

ARQUIVO_SAIDA = "amostras_validacao.h"

TARGET_FS = 16000

WINDOW_MS = 550

AMOSTRAS_NECESSARIAS = int(TARGET_FS * WINDOW_MS / 1000)

GANHO = 1.5

def processar_audio(caminho_arquivo):

    #Leitura
    fs, data = wavfile.read(caminho_arquivo)

    print("\n================================================")
    print(f"Arquivo: {os.path.basename(caminho_arquivo)}")
    print("================================================")

    print(f"Sample rate original: {fs}")
    print(f"Shape original: {data.shape}")
    print(f"Tipo original: {data.dtype}")



    if len(data.shape) > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    data = data - np.mean(data)

    #Resample 16kHz
    if fs != TARGET_FS:

        novo_tamanho = int(len(data) * TARGET_FS / fs)

        data = signal.resample(data, novo_tamanho)

        print(f"Resample aplicado: {fs} -> {TARGET_FS}")

    data = data * GANHO

    data = np.clip(data, -32768, 32767)

    energia = np.abs(data)

    pico = np.argmax(energia)

    inicio = max(
        0,
        pico - AMOSTRAS_NECESSARIAS // 2
    )

    fim = inicio + AMOSTRAS_NECESSARIAS

    data = data[inicio:fim]

    if len(data) < AMOSTRAS_NECESSARIAS:

        padding = AMOSTRAS_NECESSARIAS - len(data)

        data = np.pad(
            data,
            (0, padding),
            mode='constant'
        )

    print(f"Tamanho final: {len(data)}")

    print(f"MAX: {np.max(data):.2f}")
    print(f"MIN: {np.min(data):.2f}")

    print(f"Média: {np.mean(data):.4f}")

    data = data.astype(np.int16)

    return data

#Processa todos os áudios
audios_processados = []

for arquivo in sorted(os.listdir(PASTA_AUDIOS)):

    if arquivo.lower().endswith(".wav"):

        caminho = os.path.join(PASTA_AUDIOS, arquivo)

        dados = processar_audio(caminho)

        nome_variavel = (
            arquivo
            .replace(".wav", "")
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
        )

        audios_processados.append(
            (nome_variavel, dados)
        )

#Gera header
with open(ARQUIVO_SAIDA, "w") as f:
    f.write("#ifndef AMOSTRAS_VALIDACAO_H\n")
    f.write("#define AMOSTRAS_VALIDACAO_H\n\n")

    f.write("#include <stdint.h>\n\n")

    f.write(f"#define QUANTIDADE_AMOSTRAS {AMOSTRAS_NECESSARIAS}\n\n")

    for nome, dados in audios_processados:

        f.write(
            f"static const int16_t {nome}[QUANTIDADE_AMOSTRAS] = {{\n"
        )

        linha = "    "

        for i, valor in enumerate(dados):

            linha += f"{valor}, "

            if (i + 1) % 16 == 0:
                f.write(linha + "\n")
                linha = "    "

        if linha.strip():
            f.write(linha + "\n")

        f.write("};\n\n")

    f.write("#endif\n")

print(f"Total de áudios: {len(audios_processados)}")
#define EIDSP_QUANTIZE_FILTERBANK 0

/* Includes ---------------------------------------------------------------- */
#include <miria-project-1_inferencing.h>

#include "C:/Users/miria/Documents/UFAL/TCC/codigos/amostras_validacao.h"

#define TOTAL_DE_AUDIOS 20

static const int16_t* LISTA_DE_AMOSTRAS[TOTAL_DE_AUDIOS] = {

    /*normal_1,
    normal_2,
    normal_3,
    normal_4,
    normal_5,
    normal_6,
    normal_7,
    normal_8,
    normal_9,
    normal_10,
    normal_11,
    normal_12,
    normal_13,
    normal_14,
    normal_15,
    normal_16,
    normal_17,
    normal_18,
    normal_19,
    normal_20*/


    abnormal_1,
    abnormal_2,
    abnormal_3,
    abnormal_4,
    abnormal_5,
    abnormal_6,
    abnormal_7,
    abnormal_8,
    abnormal_9,
    abnormal_10,
    abnormal_11,
    abnormal_12,
    abnormal_13,
    abnormal_14,
    abnormal_15,
    abnormal_16,
    abnormal_17,
    abnormal_18,
    abnormal_19,
    abnormal_20
};

static const int16_t* AMOSTRA_SELECIONADA =
    LISTA_DE_AMOSTRAS[0];

static bool debug_nn = false;

static int obter_dados_audio_estatico(
    size_t offset,
    size_t length,
    float *out_ptr)
{

    //Converte int16 -> float
    numpy::int16_to_float(
        &AMOSTRA_SELECIONADA[offset],
        out_ptr,
        length
    );

    return 0;
}

void setup()
{
    Serial.begin(115200);

    while (!Serial);

    for (int i = 0; i < 30; i++) {
        Serial.println();
    }

    Serial.println("====================================");
    Serial.println("Validacao on-device");
    Serial.println("====================================");

    ei_printf(
        "EI_CLASSIFIER_FREQUENCY: %d\n",
        EI_CLASSIFIER_FREQUENCY
    );

    ei_printf(
        "EI_CLASSIFIER_RAW_SAMPLE_COUNT: %d\n",
        EI_CLASSIFIER_RAW_SAMPLE_COUNT
    );

    ei_printf(
        "QUANTIDADE_AMOSTRAS: %d\n",
        QUANTIDADE_AMOSTRAS
    );

    Serial.println("====================================");
}

void loop()
{
    // Processa todos os áudios

    for (int i = 0; i < TOTAL_DE_AUDIOS; i++) {
        AMOSTRA_SELECIONADA =
            LISTA_DE_AMOSTRAS[i];

        ei_printf("\n");
        ei_printf("====================================\n");
        ei_printf("PROCESSANDO AUDIO [%d]\n", i);
        ei_printf("====================================\n");

        for (int j = 0;
             j < QUANTIDADE_AMOSTRAS;
             j++)
        {
            int16_t v = abs(AMOSTRA_SELECIONADA[j]);
        }

        signal_t signal;

        signal.total_length = QUANTIDADE_AMOSTRAS;

        signal.get_data = &obter_dados_audio_estatico;

        
        ei_impulse_result_t result = { 0 };

        // Executa inferência

        EI_IMPULSE_ERROR r =
            run_classifier(
                &signal,
                &result,
                debug_nn
            );

        if (r != EI_IMPULSE_OK) {

            ei_printf(
                "ERRO AO RODAR MODELO (%d)\n",
                r
            );

            continue;
        }

        ei_printf(
            "\nDSP: %d ms\n",
            result.timing.dsp
        );

        ei_printf(
            "Classificacao: %d ms\n",
            result.timing.classification
        );

        //Resultados

        ei_printf("\nRESULTADOS:\n");

        for (size_t ix = 0;
             ix < EI_CLASSIFIER_LABEL_COUNT;
             ix++)
        {
            ei_printf(
                "    %s: %.5f\n",
                result.classification[ix].label,
                result.classification[ix].value
            );
        }

#if EI_CLASSIFIER_HAS_ANOMALY == 1

        ei_printf(
            "    anomaly: %.5f\n",
            result.anomaly
        );

#endif

        delay(5000);
    }

    ei_printf("\n");
    ei_printf("====================================\n");
    ei_printf("FIM DOS TESTES\n");
    ei_printf("====================================\n");

    while (1);
}
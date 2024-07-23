# Modelagem de Fidelidade de Clientes com Regressão Logística em R

Neste estudo de caso, utilizamos técnicas avançadas de regressão logística para modelar a fidelidade de clientes de uma empresa varejista. O objetivo é prever se um cliente será fiel com base em variáveis quantitativas e qualitativas. Utilizaremos a linguagem R para realizar todas as etapas do processo, desde o carregamento dos dados até a avaliação do modelo.

## Carregamento dos Dados

Primeiro, carregamos os dados da base `dados_fidelidade.RData`:


## Carregando a base de dados
load("dados_fidelidade.RData")


## Visualização e Estatísticas Univariadas

Visualizamos a base de dados e calculamos estatísticas descritivas:


library(dplyr)
library(knitr)
library(kableExtra)

## Visualizando a base de dados
dados_fidelidade %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", full_width = F, font_size = 12)

## Estatísticas Univariadas
summary(dados_fidelidade)

## Tabelas de frequência para variáveis qualitativas
table(dados_fidelidade$atendimento)
table(dados_fidelidade$sortimento)
table(dados_fidelidade$acessibilidade)
table(dados_fidelidade$preço)

## Estrutura da base de dados
glimpse(dados_fidelidade)


## Estimação de um Modelo Logístico Binário

Estimamos o modelo de regressão logística:


## Estimando o modelo de regressão logística
modelo_fidelidade <- glm(fidelidade ~ . - id, data = dados_fidelidade, family = "binomial")

## Resumo do modelo
summary(modelo_fidelidade)

## Outra forma de apresentar os resultados
library(jtools)
summ(modelo_fidelidade, confint = TRUE, digits = 3, ci.width = 0.95)
export_summs(modelo_fidelidade, scale = FALSE, digits = 6)


## Procedimento Stepwise

Aplicamos o procedimento Stepwise para seleção de variáveis:


## Procedimento Stepwise
step_fidelidade <- step(object = modelo_fidelidade, k = qchisq(p = 0.05, df = 1, lower.tail = FALSE))

## Resumo do modelo após Stepwise
summary(step_fidelidade)


## Dummização das Variáveis Qualitativas

Realizamos a dummização das variáveis qualitativas:


library(fastDummies)

## Dummizando as variáveis
fidelidade_dummies <- dummy_columns(.data = dados_fidelidade,
                                    select_columns = c("atendimento", "sortimento", "acessibilidade", "preço"),
                                    remove_selected_columns = TRUE,
                                    remove_first_dummy = TRUE)

## Visualizando a base dummizada
fidelidade_dummies %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", full_width = F, font_size = 12)


## Reestimando o Modelo com Variáveis Dummizadas

Reestimamos o modelo com as variáveis dummizadas:


## Estimando o modelo com variáveis dummizadas
modelo_fidelidade_dummies <- glm(fidelidade ~ . - id, data = fidelidade_dummies, family = "binomial")

## Resumo do modelo
summary(modelo_fidelidade_dummies)

## Log-Likelihood do modelo
logLik(modelo_fidelidade_dummies)

## Procedimento Stepwise
step_fidelidade_dummies <- step(object = modelo_fidelidade_dummies, k = qchisq(p = 0.05, df = 1, lower.tail = FALSE))

## Resumo do modelo após Stepwise
summary(step_fidelidade_dummies)

## Outra forma de apresentar os resultados
summ(step_fidelidade_dummies, confint = TRUE, digits = 3, ci.width = 0.95)
export_summs(step_fidelidade_dummies, scale = FALSE, digits = 6)

## Log-Likelihood do modelo após Stepwise
logLik(step_fidelidade_dummies)

## Comparando os modelos
library(lmtest)
lrtest(modelo_fidelidade_dummies, step_fidelidade_dummies)


## Construção de uma Matriz de Confusão

Construímos uma matriz de confusão para avaliar a performance do modelo:


library(caret)

## Construção da matriz de confusão
confusionMatrix(table(predict(step_fidelidade_dummies, type = "response") >= 0.5, dados_fidelidade$fidelidade == "sim")[2:1, 2:1])


## Igualando Especificidade e Sensitividade

Estabelecemos um critério para igualar a especificidade e a sensitividade:


library(ROCR)

## Predições
predicoes <- prediction(predictions = step_fidelidade_dummies$fitted.values, labels = dados_fidelidade$fidelidade)

## Dados da curva ROC
dados_curva_roc <- performance(predicoes, measure = "sens")
sensitividade <- dados_curva_roc@y.values[[1]]

especificidade <- performance(predicoes, measure = "spec")
especificidade <- especificidade@y.values[[1]]

cutoffs <- dados_curva_roc@x.values[[1]]

## Criando um data frame para plotagem
dados_plotagem <- cbind.data.frame(cutoffs, especificidade, sensitividade)

## Visualizando o data frame
dados_plotagem %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", full_width = F, font_size = 12)

## Plotagem
library(ggplot2)
library(plotly)

ggplotly(
  dados_plotagem %>%
    ggplot(aes(x = cutoffs, y = especificidade)) +
    geom_line(aes(color = "Especificidade"), size = 1) +
    geom_point(color = "chocolate", fill = "chocolate1", size = 1, shape = 22) +
    geom_line(aes(x = cutoffs, y = sensitividade, color = "Sensitividade"), size = 1) +
    geom_point(aes(x = cutoffs, y = sensitividade), color = "darkorchid4", size = 1) +
    labs(x = "Cutoff", y = "Sensitividade/Especificidade") +
    scale_color_manual("Legenda:", values = c("orange", "darkorchid")) +
    theme_bw()
)


## Construção da Curva ROC

Construímos a curva ROC para avaliar o desempenho do modelo:


library(pROC)

## Construção da curva ROC
ROC <- roc(response = dados_fidelidade$fidelidade, predictor = step_fidelidade_dummies$fitted.values)

## Plotagem da curva ROC
ggplotly(
  ggroc(ROC, color = "darkorchid", size = 1) +
    geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color = "orange", size = 0.2) +
    labs(x = "1 - Especificidade", y = "Sensitividade", title = paste("Área abaixo da curva:", round(ROC$auc, 3), "|", "Coeficiente de Gini", round((ROC$auc[1] - 0.5) / 0.5, 3))) +
    theme_bw()
)


## Conclusão

Este estudo de caso demonstrou como técnicas de regressão logística, junto com a dummização de variáveis qualitativas e a aplicação de procedimentos de seleção de variáveis, podem ser usadas para prever a fidelidade de clientes. Através da integração de dados, limpeza, modelagem e avaliação, fomos capazes de gerar insights valiosos para a tomada de decisões estratégicas.

### Pontos Importantes

1. **Introdução**: Fornece uma visão geral do projeto e sua importância.
2. **Carregamento dos Dados**: Descreve como os dados foram carregados.
3. **Visualização e Estatísticas Univariadas**: Detalha a visualização e as estatísticas descritivas dos dados.
4. **Estimação de um Modelo Logístico Binário**: Explica a estimação inicial do modelo de regressão logística.
5. **Procedimento Stepwise**: Descreve o procedimento de seleção de variáveis Stepwise.
6. **Dummização das Variáveis Qualitativas**: Explica a dummização das variáveis qualitativas.
7. **Reestimando o Modelo com Variáveis Dummizadas**: Detalha a reestimação do modelo com variáveis dummizadas.
8. **Construção de uma Matriz de Confusão**: Explica como construir e interpretar a matriz de confusão.
9. **Igualando Especificidade e Sensitividade**: Descreve o processo de igualar especificidade e sensitividade.
10. **Construção da Curva ROC**: Detalha a construção e interpretação da curva ROC.
11. **Conclusão**: Reflete sobre a experiência e o impacto do projeto.

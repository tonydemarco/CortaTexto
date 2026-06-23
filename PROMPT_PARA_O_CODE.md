# Prompt para o Claude Code

> Cole tudo abaixo (a partir de "Crie um aplicativo...") como mensagem para o Claude Code.

---

Crie um aplicativo desktop chamado **CortaTexto**, em **Python com Tkinter** (apenas biblioteca padrão + o pacote `anthropic`), que resume um texto usando a LLM da Anthropic (Claude), respeitando um limite de caracteres definido pelo usuário. Gere tudo em **um único arquivo** chamado `CortaTexto.py`, com a janela intitulada "CortaTexto", código limpo, comentado em português e organizado em funções/classe.

## Objetivo
Resumir um texto de qualquer tamanho preservando suas características (sentido essencial, fatos, tom e estilo do autor, sem inventar informação), reduzindo-o a um número de caracteres definido pelo usuário. A LLM não conta caracteres com precisão, então **quem conta é o Python** (a contagem é a fonte de verdade) e, quando necessário, o programa pede correções à LLM num loop até o texto caber.

## Regra de contagem
- Contagem exata = `len(texto)` puro: conta **todos** os caracteres (letras, espaços, pontuação, quebras de linha, acentos como code points Unicode).
- A LLM nunca decide o tamanho; ela só escreve. O Python mede e decide se aceita ou pede correção.

## Regra de aceitação (tolerância progressiva)
- O alvo é uma **faixa**: `[limite - tolerancia, limite]`.
- A tolerância é **progressiva, em duas rodadas**:
  - **Rodada 1:** `tolerancia` = **10** caracteres (campo editável na interface).
  - **Rodada 2:** se a primeira rodada de tentativas não conseguir cair na faixa, relaxe a tolerância para **20** caracteres (campo editável) e siga tentando. Ou seja, a faixa aceitável passa de `[limite-10, limite]` para `[limite-20, limite]`.
- O resumo é aceito assim que `len(resumo)` cair dentro da faixa vigente.
- O resultado **nunca** pode passar do limite, em nenhuma rodada.
- Ficar abaixo do limite dentro da tolerância é OK e desejável, para não pedir correções eternamente tentando bater o número exato.

## Loop de resumo e correção
1. Se o texto original já couber na faixa, retorne-o sem chamar a LLM.
2. Caso contrário, peça o **resumo inicial** à LLM (instrua o limite no prompt).
3. Meça com `len()`:
   - Se **acima do limite** → reenvie à LLM informando a contagem atual, o limite e quantos caracteres sobraram, pedindo para encurtar mantendo sentido, tom e correção gramatical.
   - Se **abaixo de `limite - tolerancia`** (curto demais) → reenvie pedindo para expandir até a faixa, recuperando detalhes relevantes já presentes, sem inventar informação nova.
   - Se **dentro da faixa** → aceite e pare.
4. Repita até `max_tentativas` (padrão = **10**, campo editável), aplicando a tolerância progressiva: comece a primeira rodada com tolerância 10; se essa rodada se esgotar sem cair na faixa, inicie a segunda rodada relaxando a tolerância para 20 e continue tentando (reaproveitando o melhor resumo obtido como ponto de partida). Sugestão: dividir `max_tentativas` entre as duas rodadas (ex.: metade na rodada 1, metade na rodada 2), mas garantir ao menos 1 tentativa em cada.
5. Ao longo do loop, guarde o **melhor candidato que cabe** no limite (o que estiver mais próximo do limite sem ultrapassá-lo).
6. **Último recurso** (raro): se após as duas rodadas ainda exceder o limite, aplique um corte limpo por palavras inteiras, terminando em `...`, garantindo caber, e mostre um aviso discreto de que houve corte automático.

## Chamada à LLM
- Use o SDK oficial: `import anthropic` e `cliente = anthropic.Anthropic(api_key=...)`.
- Modelo fixo: `claude-sonnet-4-6` (defina numa constante no topo, fácil de trocar).
- `max_tokens` suficiente (ex.: 2048), `temperature` baixa (ex.: 0.3) para resultados fiéis e estáveis.
- Use um **prompt de sistema** instruindo a LLM a responder APENAS com o texto do resumo, sem comentários, sem aspas, sem rótulos como "Resumo:".
- O resumo deve sair no **mesmo idioma do texto original**.
- Isole a chamada à LLM numa função única (ex.: `chamar_llm(sistema, usuario)`), para facilitar manutenção e troca de provedor.

## Interface (Tkinter)
Uma única janela com:
1. **Campo de texto de entrada**: caixa de texto grande e rolável (`Text` + `Scrollbar`), aceita colar texto de qualquer tamanho.
2. **Campo "Limite de caracteres"**: entrada numérica (`Entry`), validar que é inteiro positivo.
3. **Campo "Tolerância (caracteres a menos)"**: entrada numérica, padrão 10.
4. **Campo "Máx. de tentativas"**: entrada numérica, padrão 10.
5. **Campo "Chave da API"**: `Entry` com `show="*"` (oculto, estilo senha) para colar a `ANTHROPIC_API_KEY`. Como conveniência, se o campo estiver vazio, tente ler de `os.environ["ANTHROPIC_API_KEY"]`.
6. **Botão "Resumir"**.
7. **Janela de resultado**: caixa de texto rolável (somente leitura ou editável) que mostra o **texto final** depois das contagens e revisões.
8. **Indicador de contagem do resultado**: um rótulo discreto abaixo do resultado, ex.: `278 / 280 caracteres`. (Durante o processamento a interface fica limpa; o número só aparece no resultado final.)
9. **Botão "Copiar resultado"**: copia o texto final para a área de transferência.

## Comportamento da interface
- Durante o processamento, **não mostre status intermediário** (nada de "tentativa 2..."). Mantenha a tela limpa; mostre apenas o resultado final ao terminar.
- A chamada à API **não pode congelar a janela**: rode o trabalho numa `threading.Thread` separada e atualize a interface de volta na main thread (ex.: via `root.after(...)`). Enquanto processa, desabilite o botão "Resumir" e mostre um texto curto tipo "Processando..." (pode ser no título da janela ou num rótulo de status), reabilitando ao final.
- Trate erros com `tkinter.messagebox`: chave de API ausente/ inválida, limite não numérico, texto vazio, erro de rede/API. Mensagens claras em português.
- Validações antes de chamar a LLM: texto não vazio, limite inteiro > 0, tolerância inteira >= 0, tentativas inteiro >= 1.

## Organização do código
- Separe **lógica** de **interface**:
  - Funções puras testáveis: `contar(texto)`, `resumir(texto, limite, tolerancia, max_tentativas, chamar_llm, ...)` retornando um objeto/dict com `texto`, `caracteres`, `sucesso`, `tentativas`, `historico`.
  - `resumir(...)` deve aceitar a função `chamar_llm` por injeção de dependência (para testar com um mock sem chamar a API real).
  - Classe da interface (ex.: `App(tk.Tk)`) que usa essa lógica.
- Inclua um cabeçalho/docstring explicando uso e dependências (`pip install anthropic`).
- Inclua `if __name__ == "__main__":` que abre a janela.

## Entregáveis
1. `CortaTexto.py` — o aplicativo completo.
2. Ao final, escreva no chat as instruções de execução:
   - `pip install anthropic`
   - `python CortaTexto.py`
3. Sugira (sem necessariamente implementar) alguns testes com mock da LLM para validar: contagem exata com acentos/espaços, loop de correção quando acima do limite, expansão quando curto demais, aceitação dentro da faixa, e corte de segurança no último recurso.

Comente o código em português e priorize legibilidade.

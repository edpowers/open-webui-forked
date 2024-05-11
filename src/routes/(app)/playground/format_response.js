export function reformatJSON(receivedJSON) {
    const { content, done, total_duration, load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration } = receivedJSON;

    const reformattedJSON = {
      object: 'chat.completion',
      choices: [
        {
          finish_reason: done ? 'stop' : null,
          index: 0,
          message: {
            content: content.replace(/\\n/g, '\n'),
            role: 'assistant',
            logprobs: null,
          },
        },
      ],
      id: generateUniqueId(),
      created: Date.now() / 1000,
      model: 'mistral_7b_instruct_v2_quant_v2',
      usage: {
        prompt_tokens: prompt_eval_count,
        completion_tokens: eval_count,
        total_tokens: prompt_eval_count + eval_count,
      },
    };

    return reformattedJSON;
  }

  function generateUniqueId() {
    const timestamp = Date.now().toString(36);
    const randomStr = Math.random().toString(36).substring(2, 9);
    return `chatcmpl-${timestamp}-${randomStr}`;
  }
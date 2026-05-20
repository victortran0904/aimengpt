import { DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE } from '@/utils/app/const';
import { OpenAIError, OpenAIStream } from '@/utils/server';
import type { ScientificSourceManifestEntry } from '@/utils/server/scientific-evidence';

import { ChatBody, Message } from '@/types/chat';

// @ts-expect-error
import wasm from '../../node_modules/@dqbd/tiktoken/lite/tiktoken_bg.wasm?module';

import tiktokenModel from '@dqbd/tiktoken/encoders/cl100k_base.json';
import { Tiktoken, init } from '@dqbd/tiktoken/lite/init';
import { codeBlock, oneLine } from 'common-tags';

export const config = {
  runtime: 'edge',
};

type FetchDocumentsResponse = {
  evidenceContext?: string;
  sourceManifest?: ScientificSourceManifestEntry[];
};

function formatSourceManifest(
  sourceManifest: ScientificSourceManifestEntry[],
): string {
  return sourceManifest
    .map((source, index) => {
      return `${index + 1}. ${source.title} (${
        source.source
      }) -> keys: ${source.citationKeys.join(', ')}`;
    })
    .join('\n');
}

async function fetchScientificEvidence(
  req: Request,
  lastMessageContent: string,
) {
  try {
    const fetchDocumentsUrl = new URL(
      '/api/fetch-documents',
      req.url,
    ).toString();
    const response = await fetch(fetchDocumentsUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input: lastMessageContent,
        nResults: 8,
        maxEvidenceChars: 12000,
      }),
    });

    if (!response.ok) {
      throw new Error(`Error fetching documents: ${response.statusText}`);
    }

    const data = (await response.json()) as FetchDocumentsResponse;
    return {
      evidenceContext:
        typeof data.evidenceContext === 'string' ? data.evidenceContext : '',
      sourceManifest: Array.isArray(data.sourceManifest)
        ? data.sourceManifest
        : [],
    };
  } catch (error) {
    console.error('Error fetching scientific evidence:', error);
    throw error;
  }
}

const handler = async (req: Request): Promise<Response> => {
  try {
    const { model, messages, key, prompt, temperature } =
      (await req.json()) as ChatBody;

    await init((imports) => WebAssembly.instantiate(wasm, imports));
    const encoding = new Tiktoken(
      tiktokenModel.bpe_ranks,
      tiktokenModel.special_tokens,
      tiktokenModel.pat_str,
    );

    let promptToSend = codeBlock`
    ${oneLine`
      You are a very enthusiastic AI assistant  who loves
      to help people! Given the following information from
      relevant documentation, answer the user's question using
      only that information, outputted in markdown format.
    `}

    ${oneLine`
      If you are unsure
      and the answer is not explicitly written in the documentation, say
      "Sorry, I don't know how to help with that."
    `}
    
    ${oneLine`
      Always include citations from the documentation.
    `}
  `;

    if (!promptToSend) {
      promptToSend = DEFAULT_SYSTEM_PROMPT;
    }

    const lastMessage = messages[messages.length - 1];

    const { evidenceContext, sourceManifest } = await fetchScientificEvidence(
      req,
      lastMessage.content,
    );

    let temperatureToUse = temperature;
    if (temperatureToUse == null) {
      temperatureToUse = DEFAULT_TEMPERATURE;
    }

    const prompt_tokens = encoding.encode(promptToSend);

    let tokenCount = prompt_tokens.length;
    let messagesToSend: Message[] = [];

    encoding.free();

    console.log(model, promptToSend, temperatureToUse, key, messagesToSend);

    messagesToSend = [
      {
        role: 'user',
        content: codeBlock`
          Here is the evidence context:
          ${evidenceContext}
        `,
      },
      {
        role: 'user',
        content: codeBlock`
          Here is the source manifest:
          ${formatSourceManifest(sourceManifest)}
        `,
      },
      {
        role: 'user',
        content: codeBlock`
          ${oneLine`
            Answer my next question using only the above documentation.
            You must also follow the below rules when answering:
          `}
          ${oneLine`
            - Do not make up answers that are not provided in the documentation.
          `}
          ${oneLine`
            - If you are unsure and the answer is not explicitly written
            in the documentation context, say
            "Sorry, I don't know how to help with that."
          `}
          ${oneLine`
            - Prefer splitting your response into multiple paragraphs.
          `}
          ${oneLine`
            - Cite claims inline with the provided citation keys (format: [SRC-XXXXXXXX]).
          `}
          ${oneLine`
            - Only cite keys that appear in the source manifest/evidence context.
          `}
        `,
      },
      {
        role: 'user',
        content: codeBlock`
          Here is my question:
          ${oneLine`${lastMessage.content}`}
      `,
      },
    ];

    const stream = await OpenAIStream(
      model,
      promptToSend,
      0,
      key,
      messagesToSend,
    );

    return new Response(stream);
  } catch (error) {
    console.error(error);
    if (error instanceof OpenAIError) {
      return new Response('Error', { status: 500, statusText: error.message });
    } else {
      return new Response('Error', { status: 500 });
    }
  }
};

export default handler;

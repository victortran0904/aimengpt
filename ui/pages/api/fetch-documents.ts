import type { NextApiRequest, NextApiResponse } from 'next';

import { buildScientificEvidencePayload } from '@/utils/server/scientific-evidence';

import { ChromaClient, TransformersEmbeddingFunction } from 'chromadb';

const DEFAULT_RESULTS = 8;
const MAX_RESULTS = 20;
const DEFAULT_EVIDENCE_CHARS = 12000;
const MAX_EVIDENCE_CHARS = 30000;

function parseBoundedInteger(
  value: unknown,
  defaultValue: number,
  maxValue: number,
): number {
  const parsed =
    typeof value === 'number'
      ? value
      : typeof value === 'string'
      ? Number(value)
      : NaN;

  if (!Number.isFinite(parsed) || parsed <= 0) {
    return defaultValue;
  }

  return Math.min(Math.floor(parsed), maxValue);
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  try {
    if (req.method !== 'POST') {
      return res.status(405).json({ error: 'Only POST is supported' });
    }

    const client = new ChromaClient({
      path: process.env.CHROMA_PATH || 'http://chroma-server:8000',
    });

    const query = req.body.input;
    if (typeof query !== 'string' || query.trim().length === 0) {
      return res.status(400).json({ error: 'input is required' });
    }

    const nResults = parseBoundedInteger(
      req.body.nResults,
      DEFAULT_RESULTS,
      MAX_RESULTS,
    );
    const maxEvidenceChars = parseBoundedInteger(
      req.body.maxEvidenceChars,
      DEFAULT_EVIDENCE_CHARS,
      MAX_EVIDENCE_CHARS,
    );

    const embedder = new TransformersEmbeddingFunction();

    const collection = await client.getOrCreateCollection({
      name: 'default-collection',
      embeddingFunction: embedder,
    });

    const results = await collection.query({
      nResults,
      queryTexts: [query.trim()],
    });

    const evidence = buildScientificEvidencePayload(results, {
      maxEvidenceChars,
    });

    res.status(200).json({
      ...results,
      evidenceContext: evidence.evidenceContext,
      sourceManifest: evidence.sourceManifest,
      citations: evidence.citations,
    });
  } catch (error) {
    if (error instanceof Error) {
      console.error('Error message:', error.message);
      console.error('Stack trace:', error.stack);
    } else {
      console.error('Unknown error:', error);
    }
    res.status(500).json({ error: 'An unexpected error occurred :(' });
  }
}

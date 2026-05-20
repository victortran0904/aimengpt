import {
  type ChromaQueryLike,
  buildScientificEvidencePayload,
} from '@/utils/server/scientific-evidence';

import { describe, expect, it } from 'vitest';

describe('buildScientificEvidencePayload', () => {
  it('deduplicates duplicate chunks and emits stable citation keys/source manifest', () => {
    const queryResult: ChromaQueryLike = {
      ids: [['id-1', 'id-2', 'id-3']],
      documents: [['Alpha finding', 'Alpha finding', 'Beta finding']],
      metadatas: [
        [
          {
            title: 'Paper A',
            source: '/tmp/paper-a.pdf',
            page: 2,
            chunkIndex: 0,
          },
          {
            title: 'Paper A',
            source: '/tmp/paper-a.pdf',
            page: 2,
            chunkIndex: 0,
          },
          {
            title: 'Paper A',
            source: '/tmp/paper-a.pdf',
            page: 3,
            chunkIndex: 1,
          },
        ],
      ],
      distances: [[0.01, 0.01, 0.02]],
    };

    const first = buildScientificEvidencePayload(queryResult);
    const second = buildScientificEvidencePayload(queryResult);

    expect(first.citations).toHaveLength(2);
    expect(first.citations.map((citation) => citation.key)).toEqual(
      second.citations.map((citation) => citation.key),
    );
    expect(first.sourceManifest).toHaveLength(1);
    expect(first.sourceManifest[0].citationKeys).toHaveLength(2);
  });

  it('normalizes ragged metadata and falls back safely', () => {
    const queryResult: ChromaQueryLike = {
      ids: [['id-1']],
      documents: [['  Content   with   spaces  ']],
      metadatas: [
        [
          {
            title: 123,
            sourcePath: '/var/tmp/research.pdf',
            pageNumber: '5',
            chunk_index: '2',
            chunk_id: 'c-2',
          },
        ],
      ],
      distances: [[0.1]],
    };

    const payload = buildScientificEvidencePayload(queryResult);
    expect(payload.citations).toHaveLength(1);
    expect(payload.citations[0].title).toBe('123');
    expect(payload.citations[0].source).toBe('/var/tmp/research.pdf');
    expect(payload.citations[0].page).toBe(5);
    expect(payload.citations[0].chunkIndex).toBe(2);
    expect(payload.citations[0].chunkId).toBe('c-2');
    expect(payload.citations[0].content).toBe('Content with spaces');
  });

  it('bounds evidence context length and truncates safely', () => {
    const queryResult: ChromaQueryLike = {
      ids: [['id-1']],
      documents: [['A'.repeat(200)]],
      metadatas: [[{ title: 'Large Chunk', source: 'source.pdf', page: 1 }]],
      distances: [[0.2]],
    };

    const payload = buildScientificEvidencePayload(queryResult, {
      maxChunkChars: 20,
      maxEvidenceChars: 60,
    });

    expect(payload.citations[0].content.length).toBeLessThanOrEqual(20);
    expect(payload.evidenceContext.length).toBeLessThanOrEqual(60);
    expect(payload.evidenceContext.endsWith('...')).toBe(true);
  });

  it('handles ragged/null chroma arrays without throwing', () => {
    const raggedResult: ChromaQueryLike = {
      ids: [['id-1', 'id-2'], []],
      documents: [['Chunk one', null], []],
      metadatas: [[null], []],
      distances: null,
    };

    const payload = buildScientificEvidencePayload(raggedResult);
    expect(payload.citations).toHaveLength(1);
    expect(payload.citations[0].source).toBe('unknown-source');
    expect(payload.sourceManifest).toHaveLength(1);
  });
});

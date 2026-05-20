type Primitive = string | number | boolean;

type UnknownRecord = Record<string, unknown>;

export type ChromaQueryLike = {
  ids: string[][];
  documents: (string | null)[][];
  metadatas: (Record<string, unknown> | null)[][];
  distances: null | number[][];
};

export type ScientificCitation = {
  key: string;
  sourceId: string;
  title: string;
  source: string;
  page: number | null;
  chunkIndex: number | null;
  chunkId: string | null;
  documentId: string | null;
  distance: number | null;
  content: string;
};

export type ScientificSourceManifestEntry = {
  sourceId: string;
  title: string;
  source: string;
  citationKeys: string[];
  documentIds: string[];
};

export type ScientificEvidencePayload = {
  citations: ScientificCitation[];
  sourceManifest: ScientificSourceManifestEntry[];
  evidenceContext: string;
};

export type ScientificEvidenceOptions = {
  maxEvidenceChars?: number;
  maxChunkChars?: number;
};

const DEFAULT_MAX_EVIDENCE_CHARS = 12000;
const DEFAULT_MAX_CHUNK_CHARS = 1200;

const TITLE_KEYS = ['title', 'documentTitle', 'document_title', 'pdfTitle'];
const SOURCE_KEYS = [
  'sourceLabel',
  'publicSource',
  'publicIdentifier',
  'originalFilename',
  'filename',
  'fileName',
  'source',
  'sourcePath',
  'source_path',
];
const PAGE_KEYS = ['page', 'pageNumber', 'page_number'];
const CHUNK_INDEX_KEYS = ['chunkIndex', 'chunk_index'];
const CHUNK_ID_KEYS = ['chunkId', 'chunk_id'];
const DOCUMENT_ID_KEYS = ['documentId', 'document_id', 'id'];

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asPrimitive(value: unknown): Primitive | undefined {
  if (
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    return value;
  }

  return undefined;
}

function firstPrimitive(
  record: UnknownRecord | null,
  keys: string[],
): Primitive | undefined {
  if (!record) {
    return undefined;
  }

  for (const key of keys) {
    const value = asPrimitive(record[key]);
    if (value !== undefined) {
      return value;
    }
  }

  return undefined;
}

function toCleanString(value: unknown): string | null {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  return null;
}

function toCitationSource(value: unknown): string | null {
  const source = toCleanString(value);
  if (!source) {
    return null;
  }

  const [withoutQuery] = source.split(/[?#]/);
  const normalizedPath = withoutQuery.replace(/\\/g, '/');
  const pathParts = normalizedPath.split('/').filter(Boolean);

  return pathParts[pathParts.length - 1] ?? source;
}

function toNumberOrNull(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return null;
}

function normalizeMetadata(rawMetadata: unknown): {
  title: string;
  source: string;
  page: number | null;
  chunkIndex: number | null;
  chunkId: string | null;
  documentId: string | null;
} {
  const metadata = isRecord(rawMetadata) ? rawMetadata : null;

  const title =
    toCleanString(firstPrimitive(metadata, TITLE_KEYS)) ?? 'Untitled Source';
  const source =
    toCitationSource(firstPrimitive(metadata, SOURCE_KEYS)) ?? 'unknown-source';
  const page = toNumberOrNull(firstPrimitive(metadata, PAGE_KEYS));
  const chunkIndex = toNumberOrNull(firstPrimitive(metadata, CHUNK_INDEX_KEYS));
  const chunkId = toCleanString(firstPrimitive(metadata, CHUNK_ID_KEYS));
  const documentId = toCleanString(firstPrimitive(metadata, DOCUMENT_ID_KEYS));

  return {
    title,
    source,
    page,
    chunkIndex,
    chunkId,
    documentId,
  };
}

function collapseWhitespace(content: string): string {
  return content.replace(/\s+/g, ' ').trim();
}

function safeTruncate(text: string, maxChars: number): string {
  const maxLength = Math.max(0, Math.floor(maxChars));

  if (maxLength <= 0) {
    return '';
  }

  if (text.length <= maxLength) {
    return text;
  }

  const ellipsis = '...';
  if (maxLength <= ellipsis.length) {
    return ellipsis.slice(0, maxLength);
  }

  const limit = maxLength - ellipsis.length;
  const truncated = text.slice(0, limit).trimEnd();
  return `${truncated}${ellipsis}`;
}

function hashString(value: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash +=
      (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
  }

  return (hash >>> 0).toString(16).toUpperCase().padStart(8, '0');
}

function buildCitationKey(
  source: string,
  page: number | null,
  chunkIndex: number | null,
  chunkId: string | null,
  normalizedContent: string,
): string {
  const keySeed = [
    source.toLowerCase(),
    page ?? 'na',
    chunkIndex ?? 'na',
    (chunkId ?? '').toLowerCase(),
    normalizedContent,
  ].join('|');

  return `SRC-${hashString(keySeed)}`;
}

function buildSourceId(source: string, title: string): string {
  return `DOC-${hashString(`${source.toLowerCase()}|${title.toLowerCase()}`)}`;
}

export function buildScientificEvidencePayload(
  results: ChromaQueryLike,
  options: ScientificEvidenceOptions = {},
): ScientificEvidencePayload {
  const maxEvidenceChars =
    options.maxEvidenceChars ?? DEFAULT_MAX_EVIDENCE_CHARS;
  const maxChunkChars = options.maxChunkChars ?? DEFAULT_MAX_CHUNK_CHARS;

  const citations: ScientificCitation[] = [];
  const seenCitationKeys = new Set<string>();
  const sourceManifestMap = new Map<string, ScientificSourceManifestEntry>();

  const documentsByQuery = Array.isArray(results.documents)
    ? results.documents
    : [];
  const metadatasByQuery = Array.isArray(results.metadatas)
    ? results.metadatas
    : [];
  const idsByQuery = Array.isArray(results.ids) ? results.ids : [];
  const distancesByQuery = Array.isArray(results.distances)
    ? results.distances
    : [];

  for (
    let queryIndex = 0;
    queryIndex < documentsByQuery.length;
    queryIndex += 1
  ) {
    const documents = Array.isArray(documentsByQuery[queryIndex])
      ? documentsByQuery[queryIndex]
      : [];
    const metadatas = Array.isArray(metadatasByQuery[queryIndex])
      ? metadatasByQuery[queryIndex]
      : [];
    const ids = Array.isArray(idsByQuery[queryIndex])
      ? idsByQuery[queryIndex]
      : [];
    const distances = Array.isArray(distancesByQuery[queryIndex])
      ? distancesByQuery[queryIndex]
      : [];

    for (let index = 0; index < documents.length; index += 1) {
      const rawContent = documents[index];
      if (typeof rawContent !== 'string') {
        continue;
      }

      const normalizedContent = collapseWhitespace(rawContent);
      if (!normalizedContent) {
        continue;
      }

      const metadata = normalizeMetadata(metadatas[index] ?? null);
      const documentId = toCleanString(ids[index]) ?? metadata.documentId;
      const citationKey = buildCitationKey(
        metadata.source,
        metadata.page,
        metadata.chunkIndex,
        metadata.chunkId,
        normalizedContent,
      );

      if (seenCitationKeys.has(citationKey)) {
        continue;
      }
      seenCitationKeys.add(citationKey);

      const sourceId = buildSourceId(metadata.source, metadata.title);
      const distance =
        typeof distances[index] === 'number' ? distances[index] : null;

      citations.push({
        key: citationKey,
        sourceId,
        title: metadata.title,
        source: metadata.source,
        page: metadata.page,
        chunkIndex: metadata.chunkIndex,
        chunkId: metadata.chunkId,
        documentId,
        distance,
        content: safeTruncate(normalizedContent, maxChunkChars),
      });

      const existingSource = sourceManifestMap.get(sourceId);
      if (!existingSource) {
        sourceManifestMap.set(sourceId, {
          sourceId,
          title: metadata.title,
          source: metadata.source,
          citationKeys: [citationKey],
          documentIds: documentId ? [documentId] : [],
        });
      } else {
        existingSource.citationKeys.push(citationKey);
        if (documentId && !existingSource.documentIds.includes(documentId)) {
          existingSource.documentIds.push(documentId);
        }
      }
    }
  }

  const evidenceLines: string[] = [];
  let usedChars = 0;
  for (const citation of citations) {
    const headerParts = [
      `[${citation.key}]`,
      `Title: ${citation.title}`,
      `Source: ${citation.source}`,
      `Page: ${citation.page ?? 'n/a'}`,
    ];

    if (citation.chunkIndex !== null) {
      headerParts.push(`Chunk: ${citation.chunkIndex}`);
    }

    const block = `${headerParts.join(' | ')}\n${citation.content}\n`;
    if (usedChars + block.length > maxEvidenceChars) {
      const remaining = maxEvidenceChars - usedChars;
      if (remaining > 0) {
        evidenceLines.push(safeTruncate(block, remaining));
      }
      break;
    }

    evidenceLines.push(block);
    usedChars += block.length;
  }

  return {
    citations,
    sourceManifest: Array.from(sourceManifestMap.values()),
    evidenceContext: evidenceLines.join('\n'),
  };
}

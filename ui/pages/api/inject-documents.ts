import type { NextApiRequest, NextApiResponse } from 'next';

import { ChromaClient, TransformersEmbeddingFunction } from 'chromadb';
import { IncomingForm } from 'formidable';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  if (req.method !== 'POST') {
    return res.status(405).end();
  }

  const form = new IncomingForm();
  form.parse(req, async (err, fields, files) => {
    try {
      if (err) {
        return res.status(400).json({ error: 'Failed to upload file' });
      }

      const pdfFile = Array.isArray(files.pdf) ? files.pdf[0] : files.pdf;
      if (!pdfFile?.filepath) {
        return res.status(400).json({ error: 'A PDF file is required' });
      }

      const client = new ChromaClient({
        path: process.env.CHROMA_PATH || 'http://chroma-server:8000',
      });

      const loader = new PDFLoader(pdfFile.filepath);

      const originalDocs = await loader.load();

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 100,
      });

      const docs = await splitter.splitDocuments(originalDocs);

      // Process the documents and perform other logic
      const { ids, metadatas, documentContents } = processDocuments(docs);

      const embedder = new TransformersEmbeddingFunction();
      const collection = await client.getOrCreateCollection({
        name: 'default-collection',
        embeddingFunction: embedder,
      });

      await collection.add({
        ids,
        metadatas,
        documents: documentContents,
      });

      res.status(200).json({
        message: 'Documents processed successfully',
        documentCount: ids.length,
      });
    } catch (error) {
      console.error(error);
      res
        .status(500)
        .json({ message: 'An error occurred while processing the documents' });
    }
  });
}

type PrimitiveMetadata = Record<string, string | number | boolean>;

type LoadedDocument = {
  pageContent: string;
  metadata?: unknown;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function getPrimitive(
  record: Record<string, unknown>,
  key: string,
): string | number | boolean | null {
  const value = record[key];
  if (
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    return value;
  }

  return null;
}

function asNonEmptyString(value: unknown): string | null {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  return null;
}

function asNumber(value: unknown): number | null {
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

function getPageFromMetadata(metadata: Record<string, unknown>): number | null {
  const directPage = asNumber(
    getPrimitive(metadata, 'page') ?? getPrimitive(metadata, 'pageNumber'),
  );
  if (directPage !== null) {
    return directPage;
  }

  const loc = metadata.loc;
  if (!isRecord(loc)) {
    return null;
  }

  return asNumber(getPrimitive(loc, 'pageNumber') ?? getPrimitive(loc, 'page'));
}

function getPdfInfoPrimitive(
  metadata: Record<string, unknown>,
  key: string,
): string | number | boolean | null {
  const pdf = metadata.pdf;
  if (!isRecord(pdf)) {
    return null;
  }

  const info = pdf.info;
  if (!isRecord(info)) {
    return null;
  }

  return getPrimitive(info, key);
}

function processDocuments(docs: LoadedDocument[]) {
  const ids: string[] = [];
  const metadatas: PrimitiveMetadata[] = [];
  const documentContents: string[] = [];

  for (let index = 0; index < docs.length; index += 1) {
    const document = docs[index];
    const metadata = isRecord(document.metadata) ? document.metadata : {};

    const sourcePath =
      asNonEmptyString(getPrimitive(metadata, 'source')) ??
      asNonEmptyString(getPrimitive(metadata, 'sourcePath')) ??
      `document-${index + 1}.pdf`;
    const filename = path.basename(sourcePath);
    const fallbackTitle =
      filename.length > 0 ? filename : `Document ${index + 1}`;
    const titleFromMetadata =
      asNonEmptyString(getPrimitive(metadata, 'title')) ??
      asNonEmptyString(getPrimitive(metadata, 'documentTitle')) ??
      asNonEmptyString(getPdfInfoPrimitive(metadata, 'Title'));
    const title = titleFromMetadata ?? fallbackTitle;
    const page = getPageFromMetadata(metadata);
    const chunkIndex =
      asNumber(
        getPrimitive(metadata, 'chunkIndex') ??
          getPrimitive(metadata, 'chunk_index'),
      ) ?? index;

    const generatedId = uuidv4();
    const chunkId =
      asNonEmptyString(
        getPrimitive(metadata, 'chunkId') ?? getPrimitive(metadata, 'chunk_id'),
      ) ?? `${filename}:${page ?? 'na'}:${chunkIndex}`;
    const documentId =
      asNonEmptyString(
        getPrimitive(metadata, 'documentId') ??
          getPrimitive(metadata, 'document_id'),
      ) ?? generatedId;

    const metadataToStore: PrimitiveMetadata = {
      title,
      source: sourcePath,
      sourcePath,
      filename,
      chunkIndex,
      chunkId,
      documentId,
    };

    if (page !== null) {
      metadataToStore.page = page;
    }

    const optionalPdfInfoFields = [
      'Author',
      'Subject',
      'Keywords',
      'Creator',
      'Producer',
    ];
    for (const field of optionalPdfInfoFields) {
      const value = getPdfInfoPrimitive(metadata, field);
      if (value !== null) {
        metadataToStore[`pdf${field}`] = value;
      }
    }

    ids.push(generatedId);
    metadatas.push(metadataToStore);
    documentContents.push(document.pageContent);
  }

  return { ids, metadatas, documentContents };
}

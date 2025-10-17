import { PipeError } from './types';

export class PipeApiError extends Error implements PipeError {
  public status?: number;
  public code?: string;

  constructor(message: string, status?: number, code?: string) {
    super(message);
    this.name = 'PipeApiError';
    this.status = status;
    this.code = code;
  }

  static fromResponse(response: any): PipeApiError {
    const message = response.data?.message || response.statusText || 'Unknown API error';
    return new PipeApiError(message, response.status, response.data?.code);
  }
}

export class PipeValidationError extends Error implements PipeError {
  constructor(message: string) {
    super(message);
    this.name = 'PipeValidationError';
  }
}

export class PipeSessionError extends Error implements PipeError {
  constructor(message: string) {
    super(message);
    this.name = 'PipeSessionError';
  }
}

export class PipeStorageError extends Error implements PipeError {
  constructor(message: string) {
    super(message);
    this.name = 'PipeStorageError';
  }
}
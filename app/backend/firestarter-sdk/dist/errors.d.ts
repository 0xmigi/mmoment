import { PipeError } from './types';
export declare class PipeApiError extends Error implements PipeError {
    status?: number;
    code?: string;
    constructor(message: string, status?: number, code?: string);
    static fromResponse(response: any): PipeApiError;
}
export declare class PipeValidationError extends Error implements PipeError {
    constructor(message: string);
}
export declare class PipeSessionError extends Error implements PipeError {
    constructor(message: string);
}
export declare class PipeStorageError extends Error implements PipeError {
    constructor(message: string);
}
//# sourceMappingURL=errors.d.ts.map
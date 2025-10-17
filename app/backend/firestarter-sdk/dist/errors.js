"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.PipeStorageError = exports.PipeSessionError = exports.PipeValidationError = exports.PipeApiError = void 0;
class PipeApiError extends Error {
    constructor(message, status, code) {
        super(message);
        this.name = 'PipeApiError';
        this.status = status;
        this.code = code;
    }
    static fromResponse(response) {
        const message = response.data?.message || response.statusText || 'Unknown API error';
        return new PipeApiError(message, response.status, response.data?.code);
    }
}
exports.PipeApiError = PipeApiError;
class PipeValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = 'PipeValidationError';
    }
}
exports.PipeValidationError = PipeValidationError;
class PipeSessionError extends Error {
    constructor(message) {
        super(message);
        this.name = 'PipeSessionError';
    }
}
exports.PipeSessionError = PipeSessionError;
class PipeStorageError extends Error {
    constructor(message) {
        super(message);
        this.name = 'PipeStorageError';
    }
}
exports.PipeStorageError = PipeStorageError;
//# sourceMappingURL=errors.js.map
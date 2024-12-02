// src/services/storage-service.ts
const DELETED_HASHES_KEY = 'deleted_image_hashes';

export const storageService = {
  addDeletedHash(hash: string) {
    const deleted = this.getDeletedHashes();
    deleted.add(hash);
    localStorage.setItem(DELETED_HASHES_KEY, JSON.stringify([...deleted]));
  },

  getDeletedHashes(): Set<string> {
    const stored = localStorage.getItem(DELETED_HASHES_KEY);
    return new Set(stored ? JSON.parse(stored) : []);
  },

  clearDeletedHashes() {
    localStorage.removeItem(DELETED_HASHES_KEY);
  }
};
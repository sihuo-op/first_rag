import request from '@/utils/request'

export function getDocuments(params) {
  return request.get('/v1/documents', { params })
}

export function getDocument(id) {
  return request.get(`/v1/documents/${id}`)
}

export function uploadDocument(file, title, onUploadProgress) {
  const formData = new FormData()
  formData.append('file', file)
  if (title) {
    formData.append('title', title)
  }
  return request.post('/v1/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress
  })
}

export function deleteDocument(id) {
  return request.delete(`/v1/documents/${id}`)
}

export function updateDocument(id, data) {
  return request.put(`/v1/documents/${id}`, data)
}

export function getDocumentStatus(id) {
  return request.get(`/v1/documents/${id}/status`)
}

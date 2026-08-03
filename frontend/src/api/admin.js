import request from '@/utils/request'

export function getUsers(params) {
  return request.get('/v1/admin/users', { params })
}

export function getUser(id) {
  return request.get(`/v1/admin/users/${id}`)
}

export function updateUser(id, data) {
  return request.put(`/v1/admin/users/${id}`, data)
}

export function deleteUser(id) {
  return request.delete(`/v1/admin/users/${id}`)
}

export function getAllDocuments(params) {
  return request.get('/v1/admin/documents', { params })
}

export function adminDeleteDocument(id) {
  return request.delete(`/v1/admin/documents/${id}`)
}

export function getStats() {
  return request.get('/v1/admin/stats')
}

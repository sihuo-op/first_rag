import request from '@/utils/request'

export function login(data) {
  return request.post('/v1/auth/login', data)
}

export function register(data) {
  return request.post('/v1/auth/register', data)
}

export function getCurrentUser() {
  return request.get('/v1/auth/me')
}

export function refreshToken(refreshToken) {
  return request.post('/v1/auth/refresh', null, {
    params: { refresh_token: refreshToken }
  })
}

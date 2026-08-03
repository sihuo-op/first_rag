import request from '@/utils/request'
import { getToken } from '@/utils/auth'

export function getConversations(params) {
  return request.get('/v1/conversations', { params })
}

export function createConversation(data) {
  return request.post('/v1/conversations', data)
}

export function getConversation(id) {
  return request.get(`/v1/conversations/${id}`)
}

export function updateConversation(id, data) {
  return request.put(`/v1/conversations/${id}`, data)
}

export function deleteConversation(id) {
  return request.delete(`/v1/conversations/${id}`)
}

export function getMessages(id, params) {
  return request.get(`/v1/conversations/${id}/messages`, { params })
}

export function chat(data) {
  return request.post('/v1/chat', data)
}

export async function streamChat(data, handlers = {}) {
  const response = await fetch('/api/v1/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(getToken() ? { Authorization: `Bearer ${getToken()}` } : {})
    },
    body: JSON.stringify({ ...data, stream: true })
  })

  if (!response.ok) {
    let message = `请求失败: ${response.status}`
    try {
      const errorData = await response.json()
      message = errorData.detail || message
    } catch {
    }
    throw new Error(message)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const handleEvent = (rawEvent) => {
    const dataLines = rawEvent
      .split('\n')
      .filter(line => line.startsWith('data:'))
      .map(line => line.slice(5).trimStart())

    if (dataLines.length === 0) return

    const payloadText = dataLines.join('\n')
    if (payloadText === '[DONE]') return

    const event = JSON.parse(payloadText)
    if (event.type === 'status') handlers.onStatus?.(event)
    else if (event.type === 'content') handlers.onContent?.(event)
    else if (event.type === 'done') handlers.onDone?.(event)
    else if (event.type === 'error') handlers.onError?.(event)
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() || ''

    for (const part of parts) {
      if (part.trim()) handleEvent(part)
    }
  }

  if (buffer.trim()) handleEvent(buffer)
}

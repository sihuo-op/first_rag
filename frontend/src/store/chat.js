import { defineStore } from 'pinia'
import { getConversations, createConversation, deleteConversation, getMessages, chat, streamChat, sendFeedback } from '@/api/chat'

export const useChatStore = defineStore('chat', {
  state: () => ({
    conversations: [],
    currentConversationId: null,
    messages: [],
    loading: false
  }),

  actions: {
    async fetchConversations() {
      this.conversations = await getConversations()
      // 如果没有当前对话，尝试恢复上次对话
      if (!this.currentConversationId && this.conversations.length > 0) {
        const lastConvId = localStorage.getItem('lastConversationId')
        if (lastConvId) {
          const exists = this.conversations.find(c => c.id === parseInt(lastConvId))
          if (exists) {
            await this.selectConversation(parseInt(lastConvId))
          }
        }
      }
    },

    async createConversation(title) {
      const conv = await createConversation({ title })
      this.conversations.unshift(conv)
      return conv
    },

    async deleteConversation(id) {
      await deleteConversation(id)
      this.conversations = this.conversations.filter(c => c.id !== id)
      if (this.currentConversationId === id) {
        this.currentConversationId = null
        this.messages = []
        localStorage.removeItem('lastConversationId')
      }
    },

    async selectConversation(id) {
      this.currentConversationId = id
      localStorage.setItem('lastConversationId', id)
      this.messages = await getMessages(id)
    },

    async sendMessage(query, useRag = true) {
      const now = Date.now()
      const userTempId = 'temp-user-' + now
      const assistantTempId = 'temp-assistant-' + now

      this.messages.push({
        id: userTempId,
        role: 'user',
        content: query,
        created_at: new Date().toISOString()
      })

      this.messages.push({
        id: assistantTempId,
        role: 'assistant',
        content: '',
        created_at: new Date().toISOString(),
        is_streaming: true,
        streaming_status: '正在准备回答...'
      })

      const updateAssistant = (patch) => {
        const index = this.messages.findIndex(m => m.id === assistantTempId)
        if (index >= 0) {
          this.messages[index] = { ...this.messages[index], ...patch }
        }
      }

      this.loading = true
      try {
        let finalResponse = null
        await streamChat({
          query,
          conversation_id: this.currentConversationId,
          use_rag: useRag
        }, {
          onStatus: (event) => {
            updateAssistant({ streaming_status: event.message })
          },
          onReasoning: (event) => {
            const current = this.messages.find(m => m.id === assistantTempId)
            updateAssistant({
              reasoning: `${current?.reasoning || ''}${event.content || ''}`,
              streaming_status: '正在思考...'
            })
          },
          onContent: (event) => {
            const current = this.messages.find(m => m.id === assistantTempId)
            updateAssistant({
              content: `${current?.content || ''}${event.content || ''}`,
              streaming_status: '正在生成答案...'
            })
          },
          onError: (event) => {
            updateAssistant({ streaming_status: event.message || '处理过程中出现错误' })
          },
          onDone: (event) => {
            finalResponse = event
            updateAssistant({
              id: event.message_id,
              is_streaming: false,
              streaming_status: '',
              process_time: event.process_time,
              debug_info: event.debug_info,
              agentic_info: event.agentic_info,
              retrieved_chunks: event.retrieved_chunks || []
            })
          }
        })

        if (finalResponse?.conversation_id) {
          if (!this.currentConversationId) {
            this.currentConversationId = finalResponse.conversation_id
            localStorage.setItem('lastConversationId', finalResponse.conversation_id)
          }
          await this.fetchConversations()
        }

        return finalResponse
      } catch (e) {
        updateAssistant({
          is_streaming: false,
          streaming_status: e.message || '请求失败',
          content: '抱歉，请求失败，请稍后重试。'
        })
        throw e
      } finally {
        this.loading = false
      }
    },

    clearCurrentChat() {
      this.currentConversationId = null
      this.messages = []
      localStorage.removeItem('lastConversationId')
    },

    async sendFeedback(messageId, polarity) {
      const idx = this.messages.findIndex(m => m.id === messageId)
      const prev = idx >= 0 ? this.messages[idx].feedback_polarity : null
      if (idx >= 0) {
        this.messages[idx].feedback_polarity = polarity
      }
      try {
        await sendFeedback({ message_id: messageId, polarity })
        return { ok: true }
      } catch (e) {
        if (idx >= 0) {
          this.messages[idx].feedback_polarity = prev
        }
        return { ok: false, error: e.message || '反馈提交失败' }
      }
    }
  }
})

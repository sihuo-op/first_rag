import { defineStore } from 'pinia'
import { getToken, setToken, removeToken, getUser, setUser, removeUser } from '@/utils/auth'
import { login, register, getCurrentUser } from '@/api/auth'

export const useUserStore = defineStore('user', {
  state: () => ({
    token: getToken(),
    user: getUser()
  }),

  getters: {
    isLoggedIn: (state) => !!state.token,
    isAdmin: (state) => state.user?.role === 'admin'
  },

  actions: {
    async login(credentials) {
      const response = await login(credentials)
      this.token = response.access_token
      setToken(response.access_token)
      await this.fetchCurrentUser()
    },

    async register(data) {
      return await register(data)
    },

    async fetchCurrentUser() {
      const user = await getCurrentUser()
      this.user = user
      setUser(user)
    },

    logout() {
      this.token = null
      this.user = null
      removeToken()
      removeUser()
    }
  }
})

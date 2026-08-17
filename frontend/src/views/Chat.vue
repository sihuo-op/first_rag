<template>
  <div class="chat-container">
    <el-container>
      <el-aside width="280px">
        <div class="aside-header">
          <el-button type="primary" @click="createNewChat" style="width: 100%">
            <el-icon><Plus /></el-icon>
            新对话
          </el-button>
        </div>
        <el-scrollbar class="conversation-list">
          <div
            v-for="conv in chatStore.conversations"
            :key="conv.id"
            :class="['conversation-item', { active: chatStore.currentConversationId === conv.id }]"
            @click="selectConversation(conv.id)"
          >
            <div class="conversation-info">
              <div class="conversation-title">{{ conv.title || '新对话' }}</div>
              <div class="conversation-time">{{ formatTime(conv.updated_at || conv.created_at) }}</div>
            </div>
            <el-button
              class="delete-btn"
              type="danger"
              text
              size="small"
              @click.stop="deleteConversation(conv.id)"
            >
              <el-icon><Delete /></el-icon>
            </el-button>
          </div>
        </el-scrollbar>
      </el-aside>

      <el-main class="chat-main">
        <el-scrollbar ref="messagesScrollbar" class="messages-container">
          <div v-if="chatStore.messages.length === 0" class="empty-state">
            <el-empty description="开始一段新对话吧" />
          </div>
          <div v-else class="messages-list">
            <div v-for="msg in chatStore.messages" :key="msg.id" :class="['message', msg.role]">
              <div class="message-avatar">
                <el-icon v-if="msg.role === 'user'"><User /></el-icon>
                <el-icon v-else><ChatDotRound /></el-icon>
              </div>
              <div class="message-content">
                <div class="message-bubble" v-html="formatMessage(msg.content || (getThinkingText(msg) ? '' : msg.streaming_status) || '')"></div>
                <div v-if="msg.role === 'assistant' && getThinkingText(msg)" class="thinking-block">
                  <details :open="!!msg.is_streaming">
                    <summary>🤔 思考过程 <span class="thinking-count">{{ getThinkingText(msg).length }}字</span></summary>
                    <div class="thinking-content">{{ getThinkingText(msg) }}</div>
                  </details>
                </div>
                <div v-if="msg.streaming_status && msg.role === 'assistant'" class="streaming-status">
                  {{ msg.streaming_status }}
                </div>
                <div v-if="msg.process_time && msg.role === 'assistant'" class="process-time">
                  耗时: {{ formatProcessTime(msg.process_time) }}
                </div>
                <div v-if="canShowFeedback(msg)" class="feedback-bar">
                  <button
                    class="feedback-btn"
                    :class="{ active: msg.feedback_polarity === 'up' }"
                    @click="handleFeedback(msg, 'up')"
                    title="回答有用"
                  >👍</button>
                  <button
                    class="feedback-btn"
                    :class="{ active: msg.feedback_polarity === 'down' }"
                    @click="handleFeedback(msg, 'down')"
                    title="回答有问题"
                  >👎</button>
                </div>
                <div v-if="msg.debug_info && msg.role === 'assistant'" class="debug-info">
                  <el-collapse>
                    <el-collapse-item>
                      <template #title>
                        <span class="debug-toggle">🔍 {{ isAgenticMessage(msg) ? 'Agentic' : 'RAG' }} 调试信息</span>
                      </template>

                      <!-- Agentic RAG 调试信息 -->
                      <div v-if="isAgenticMessage(msg)" class="debug-section">
                        <div class="debug-title">🔄 Agentic RAG 执行过程</div>
                        <div class="debug-content">
                          <div class="debug-stats">
                            <span>模式: {{ getAgenticInfo(msg).mode || msg.debug_info?.mode || '-' }}</span>
                            <span>尝试次数: {{ getAgenticInfo(msg).attempt_count || groupStepsByIteration(msg.debug_info?.retrieval_steps || []).length || 0 }}</span>
                            <span>置信度: {{ (((getAgenticInfo(msg).confidence || 0) * 100)).toFixed(1) }}%</span>
                            <span>最终片段: {{ msg.debug_info?.total_chunks_retrieved || 0 }}</span>
                            <span>候选片段: {{ msg.debug_info?.candidate_chunks_retrieved || getCandidateDocuments(msg).length || 0 }}</span>
                          </div>
                          <div v-if="getCandidateDocuments(msg).length" class="query-history">
                            <strong>候选检索片段:</strong>
                            <div v-for="(doc, idx) in getCandidateDocuments(msg).slice(0, 3)" :key="idx" class="query-item">
                              {{ idx + 1 }}. 分数 {{ formatScore(doc.score || doc.rerank_score || doc.rrf_score || 0) }}：{{ doc.content }}
                            </div>
                          </div>
                          <div v-if="msg.debug_info?.memory_info" class="query-history">
                            <strong>记忆改写:</strong>
                            <div class="query-item">原始问题：{{ msg.debug_info.original_query }}</div>
                            <div class="query-item">独立问题：{{ msg.debug_info.memory_info.standalone_query || msg.debug_info.rewritten_query || '-' }}</div>
                            <div class="query-item">未压缩历史数：{{ msg.debug_info.memory_info.unsummarized_message_count || 0 }}</div>
                            <div class="query-item">长期记忆数：{{ msg.debug_info.memory_info.long_term_memories?.length || 0 }}</div>
                          </div>
                          <div v-if="getAgenticInfo(msg).decomposed_tasks?.length" class="query-history">
                            <strong>子任务拆分:</strong>
                            <div v-for="task in getAgenticInfo(msg).decomposed_tasks" :key="task.id" class="query-item">
                              {{ task.id }}. {{ task.question }}
                            </div>
                          </div>
                          <div v-if="getAgenticInfo(msg).sub_tasks?.length" class="query-history">
                            <strong>子任务执行:</strong>
                            <div v-for="task in getAgenticInfo(msg).sub_tasks" :key="task.task_id" class="query-item">
                              任务 {{ task.task_id }}：{{ task.errors?.length ? '失败' : '成功' }}，置信度 {{ task.confidence || 0 }}，耗时 {{ task.elapsed_time || 0 }}s
                            </div>
                          </div>
                          <div v-if="getAgenticInfo(msg).query_history?.length > 1" class="query-history">
                            <strong>查询演变:</strong>
                            <div v-for="(q, idx) in getAgenticInfo(msg).query_history" :key="idx" class="query-item">
                              {{ idx + 1 }}. {{ q }}
                            </div>
                          </div>
                        </div>

                        <!-- 每次迭代的详细信息 -->
                        <div v-if="msg.debug_info.retrieval_steps?.length > 0" class="execution-log">
                          <div class="debug-title">📋 迭代详情</div>
                          
                          <!-- 分组显示执行日志 -->
                          <div v-for="(iterGroup, groupIdx) in groupStepsByIteration(msg.debug_info.retrieval_steps)" :key="groupIdx" class="iteration-group">
                            <div class="iteration-group-header">
                              <span class="iteration-badge">第 {{ groupIdx + 1 }} 次迭代</span>
                              <span v-if="getAgenticInfo(msg).execution_log?.[groupIdx]" class="iteration-info">
                                <span v-if="getAgenticInfo(msg).execution_log[groupIdx].confidence !== undefined">
                                  置信度: {{ getAgenticInfo(msg).execution_log[groupIdx].confidence }}
                                </span>
                                <span v-if="getAgenticInfo(msg).execution_log[groupIdx].is_sufficient !== undefined" 
                                      :class="['sufficient-badge', getAgenticInfo(msg).execution_log[groupIdx].is_sufficient ? 'pass' : 'fail']">
                                  {{ getAgenticInfo(msg).execution_log[groupIdx].is_sufficient ? '✓ 通过' : '✗ 未通过' }}
                                </span>
                              </span>
                            </div>
                            
                            <!-- 当前迭代的查询信息 -->
                            <div v-if="msg.debug_info.query_history?.[groupIdx]" class="iteration-query">
                              <span class="detail-label">查询:</span>
                              <span class="detail-value">{{ msg.debug_info.query_history[groupIdx] }}</span>
                            </div>
                            
                            <!-- 当前迭代的检索步骤（可展开） -->
                            <div class="iteration-steps">
                              <div v-for="(step, stepIdx) in iterGroup" :key="stepIdx" class="step-item">
                                <div class="step-header" @click="toggleStep(msg.id, groupIdx, stepIdx)">
                                  <span class="step-icon">{{ getStepIcon(step.step) }}</span>
                                  <span class="step-name">{{ step.desc }}</span>
                                  <span v-if="step.count" class="step-count">({{ step.count }}条)</span>
                                  <span v-if="step.time_s" class="step-time">{{ step.time_s }}s</span>
                                  <span class="step-toggle">{{ stepExpanded[msg.id + '_' + groupIdx + '_' + stepIdx] ? '▼' : '▶' }}</span>
                                </div>
                                <!-- 步骤详细结果 -->
                                <div v-show="stepExpanded[msg.id + '_' + groupIdx + '_' + stepIdx]" class="step-details">
                                  <!-- 密集向量检索结果 -->
                                  <div v-if="step.step === 'dense_search' && msg.debug_info.detail?.dense_by_type?.small" class="step-results">
                                    <div v-for="(result, resIdx) in msg.debug_info.detail.dense_by_type.small.slice(groupIdx * 4, (groupIdx + 1) * 4)" :key="resIdx" class="result-item">
                                      <span class="result-score">分数: {{ result.dense_score }}</span>
                                      <span class="result-content">{{ result.content }}</span>
                                    </div>
                                  </div>
                                  <!-- 稀疏检索结果 -->
                                  <div v-if="step.step === 'sparse_search' && msg.debug_info.detail?.sparse_results" class="step-results">
                                    <div v-for="(result, resIdx) in msg.debug_info.detail.sparse_results.slice(groupIdx * 4, (groupIdx + 1) * 4)" :key="resIdx" class="result-item">
                                      <span class="result-score">分数: {{ result.sparse_score }}</span>
                                      <span class="result-content">{{ result.content }}</span>
                                    </div>
                                  </div>
                                  <!-- RRF融合结果 -->
                                  <div v-if="step.step === 'merge' && msg.debug_info.detail?.merged_results" class="step-results">
                                    <div v-for="(result, resIdx) in msg.debug_info.detail.merged_results.slice(groupIdx * 7, (groupIdx + 1) * 7)" :key="resIdx" class="result-item">
                                      <span class="result-score">RRF: {{ result.rrf_score }}</span>
                                      <span class="result-content">{{ result.content }}</span>
                                    </div>
                                  </div>
                                  <!-- 去重结果 -->
                                  <div v-if="step.step === 'dedup' && msg.debug_info.detail?.deduped_results" class="step-results">
                                    <div v-for="(result, resIdx) in msg.debug_info.detail.deduped_results.slice(groupIdx * 4, (groupIdx + 1) * 4)" :key="resIdx" class="result-item">
                                      <span class="result-score">分数: {{ result.score || result.dense_score }}</span>
                                      <span class="result-content">{{ result.content }}</span>
                                    </div>
                                  </div>
                                  <!-- 重排序结果 -->
                                  <div v-if="step.step === 'rerank' && msg.debug_info.detail?.reranked_results" class="step-results">
                                    <div v-for="(result, resIdx) in msg.debug_info.detail.reranked_results.slice(groupIdx * 4, (groupIdx + 1) * 4)" :key="resIdx" class="result-item">
                                      <span class="result-score">重排序: {{ result.rerank_score }}</span>
                                      <span class="result-content">{{ result.content }}</span>
                                    </div>
                                  </div>
                                  <!-- 向量化信息 -->
                                  <div v-if="step.step === 'embedding'" class="step-info">
                                    <span>向量维度: {{ step.vector_dim }}</span>
                                  </div>
                                </div>
                              </div>
                            </div>
                            
                            <!-- 当前迭代的执行日志信息 -->
                            <div v-if="getAgenticInfo(msg).execution_log?.[groupIdx]" class="iteration-details">
                              <div class="detail-row" v-if="getAgenticInfo(msg).execution_log[groupIdx].new_docs_count">
                                <span class="detail-label">新增文档:</span>
                                <span class="detail-value">{{ getAgenticInfo(msg).execution_log[groupIdx].new_docs_count }} 条</span>
                              </div>
                              <div class="detail-row" v-if="getAgenticInfo(msg).execution_log[groupIdx].rewrite_type">
                                <span class="detail-label">改写策略:</span>
                                <span class="detail-value">{{ getAgenticInfo(msg).execution_log[groupIdx].rewrite_type }}</span>
                                <span v-if="getAgenticInfo(msg).execution_log[groupIdx].rewrite_time" class="detail-time">({{ getAgenticInfo(msg).execution_log[groupIdx].rewrite_time }}s)</span>
                              </div>
                              <!-- 评估信息 -->
                              <div v-if="getAgenticInfo(msg).execution_log[groupIdx].evaluation" class="evaluation-section">
                                <div class="section-title">📊 评估结果</div>
                                <div class="evaluation-item">
                                  <span class="detail-label">置信度:</span>
                                  <span class="detail-value">{{ getAgenticInfo(msg).execution_log[groupIdx].evaluation.confidence }}</span>
                                </div>
                                <div class="evaluation-item">
                                  <span class="detail-label">评估理由:</span>
                                  <span class="detail-value">{{ getAgenticInfo(msg).execution_log[groupIdx].evaluation.reason }}</span>
                                </div>
                                <div class="evaluation-item">
                                  <span class="detail-label">建议:</span>
                                  <span class="detail-value">{{ getAgenticInfo(msg).execution_log[groupIdx].evaluation.suggestion }}</span>
                                </div>
                                <div class="evaluation-item" v-if="getAgenticInfo(msg).execution_log[groupIdx].evaluation.time_s">
                                  <span class="detail-label">评估用时:</span>
                                  <span class="detail-value">{{ getAgenticInfo(msg).execution_log[groupIdx].evaluation.time_s }}s</span>
                                </div>
                              </div>
                              <!-- 发送给大模型的内容 -->
                              <div v-if="getAgenticInfo(msg).execution_log[groupIdx].llm_prompt" class="llm-section">
                                <div class="section-title">📤 发送给大模型的内容</div>
                                <pre class="llm-prompt">{{ getAgenticInfo(msg).execution_log[groupIdx].llm_prompt }}</pre>
                              </div>
                              <!-- 大模型生成的答案 -->
                              <div v-if="getAgenticInfo(msg).execution_log[groupIdx].llm_response" class="llm-section">
                                <div class="section-title">🤖 大模型响应</div>
                                <div v-if="getAgenticInfo(msg).execution_log[groupIdx].generation_time" class="generation-time">
                                  <span>生成耗时: {{ getAgenticInfo(msg).execution_log[groupIdx].generation_time }}s</span>
                                </div>
                                <div class="llm-response">{{ getAgenticInfo(msg).execution_log[groupIdx].llm_response }}</div>
                              </div>
                            </div>
                          </div>
                        </div>




                      </div>
                    </el-collapse-item>
                  </el-collapse>
                </div>
              </div>
            </div>
          </div>
          <div v-if="chatStore.loading" class="message assistant">
            <div class="message-avatar">
              <el-icon><ChatDotRound /></el-icon>
            </div>
            <div class="message-content">
              <div class="message-bubble typing">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
              </div>
            </div>
          </div>
        </el-scrollbar>

        <div class="input-area">
          <div class="input-options">
            <el-checkbox v-model="useRag">使用知识库检索</el-checkbox>
          </div>
          <div class="input-wrapper">
            <el-input
              v-model="inputText"
              type="textarea"
              :rows="3"
              placeholder="输入您的问题..."
              @keyup.enter.ctrl="sendMessage"
              :disabled="chatStore.loading"
            />
            <el-button
              type="primary"
              :icon="Promotion"
              @click="sendMessage"
              :loading="chatStore.loading"
            >
              发送
            </el-button>
          </div>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Delete, User, ChatDotRound, Promotion } from '@element-plus/icons-vue'
import { marked } from 'marked'
import { useChatStore } from '@/store/chat'

const chatStore = useChatStore()
const messagesScrollbar = ref(null)
const inputText = ref('')
const useRag = ref(true)
const detailExpanded = ref({})
const stepExpanded = ref({})

onMounted(async () => {
  await chatStore.fetchConversations()
})

function toggleDetail(msgId, type) {
  const key = `${msgId}_${type}`
  detailExpanded.value[key] = !detailExpanded.value[key]
}

function toggleStep(msgId, groupIdx, stepIdx) {
  const key = `${msgId}_${groupIdx}_${stepIdx}`
  stepExpanded.value[key] = !stepExpanded.value[key]
}

watch(() => chatStore.messages, () => {
  nextTick(() => {
    scrollToBottom()
  })
}, { deep: true })

function scrollToBottom() {
  if (messagesScrollbar.value) {
    const wrap = messagesScrollbar.value.$el.querySelector('.el-scrollbar__wrap')
    if (wrap) {
      wrap.scrollTop = wrap.scrollHeight
    }
  }
}

async function createNewChat() {
  chatStore.clearCurrentChat()
  inputText.value = ''
}

async function selectConversation(id) {
  await chatStore.selectConversation(id)
}

async function deleteConversation(id) {
  try {
    await ElMessageBox.confirm('确定要删除这个对话吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await chatStore.deleteConversation(id)
  } catch {
  }
}

async function sendMessage() {
  if (!inputText.value.trim() || chatStore.loading) return

  const query = inputText.value.trim()
  inputText.value = ''

  try {
    await chatStore.sendMessage(query, useRag.value)
  } catch (e) {
    console.error(e)
  }
}

function formatMessage(content) {
  return marked.parse(content)
}

function formatTime(timeStr) {
  if (!timeStr) return ''
  const date = new Date(timeStr)
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const hour = String(date.getHours()).padStart(2, '0')
  const minute = String(date.getMinutes()).padStart(2, '0')
  const second = String(date.getSeconds()).padStart(2, '0')
  return `${year}年${month}月${day}日 ${hour}:${minute}:${second}`
}

function formatProcessTime(ms) {
  // 兼容秒和毫秒两种格式
  const seconds = ms > 1000 ? ms / 1000 : ms
  return seconds.toFixed(2) + 's'
}

function getThinkingText(msg) {
  return msg?.reasoning || msg?.debug_info?.reasoning || ''
}

function canShowFeedback(msg) {
  return msg.role === 'assistant'
    && !msg.is_streaming
    && typeof msg.id === 'number'
}

async function handleFeedback(msg, polarity) {
  if (msg.feedback_polarity === polarity) return
  const result = await chatStore.sendFeedback(msg.id, polarity)
  if (!result.ok) {
    ElMessage.error(result.error || '反馈提交失败')
  }
}

// 判断是否为 Agentic 模式消息
function isAgenticMessage(msg) {
  const info = getAgenticInfo(msg)
  const mode = info?.mode || msg.debug_info?.mode
  if (['agentic', 'react_commander', 'parallel_rag_tools'].includes(mode)) {
    return true
  }
  if (info?.decomposed_tasks?.length || info?.sub_tasks?.length) {
    return true
  }
  if (msg.debug_info?.retrieval_steps?.length > 0) {
    return true
  }
  return false
}

// 获取 Agentic 信息（兼容即时响应和数据库加载）
function getAgenticInfo(msg) {
  return msg.agentic_info || msg.debug_info?.agentic_info || {}
}

function getCandidateDocuments(msg) {
  return getAgenticInfo(msg).candidate_documents || msg.retrieved_chunks || []
}

function formatScore(score) {
  const value = Number(score || 0)
  return Number.isFinite(value) ? value.toFixed(3) : '0.000'
}

// 将检索步骤按迭代次数分组
function groupStepsByIteration(steps) {
  if (!steps || steps.length === 0) return []
  
  const groups = []
  let currentGroup = []
  const cycleLength = 6 // embedding, dense_search, sparse_search, merge, dedup, rerank
  
  steps.forEach((step, index) => {
    if (step.step === 'embedding' && index > 0) {
      groups.push([...currentGroup])
      currentGroup = []
    }
    currentGroup.push(step)
  })
  
  if (currentGroup.length > 0) {
    groups.push(currentGroup)
  }
  
  return groups
}

// 获取步骤对应的图标
function getStepIcon(stepType) {
  const icons = {
    'embedding': '🧠',
    'dense_search': '🔍',
    'sparse_search': '📋',
    'merge': '🔗',
    'dedup': '🗑️',
    'rerank': '📊'
  }
  return icons[stepType] || '⚙️'
}
</script>

<style scoped>
.chat-container {
  height: 100%;
}

.chat-container .el-container {
  height: 100%;
}

.aside-header {
  padding: 15px;
  border-bottom: 1px solid #e6e6e6;
}

.conversation-list {
  height: calc(100% - 70px);
}

.conversation-item {
  padding: 12px 15px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #f0f0f0;
}

.conversation-item:hover {
  background-color: #f5f7fa;
}

.conversation-item.active {
  background-color: #e6f7ff;
}

.conversation-info {
  flex: 1;
  overflow: hidden;
}

.conversation-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 14px;
}

.conversation-time {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

.delete-btn {
  opacity: 0;
}

.conversation-item:hover .delete-btn {
  opacity: 1;
}

.chat-main {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 0;
}

.messages-container {
  flex: 1;
  overflow: auto;
}

.empty-state {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
}

.messages-list {
  padding: 20px;
}

.message {
  display: flex;
  margin-bottom: 20px;
}

.message.user {
  flex-direction: row-reverse;
}

.message-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: #409eff;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 20px;
  flex-shrink: 0;
}

.message.user .message-avatar {
  background-color: #67c23a;
}

.message-content {
  max-width: 70%;
  margin: 0 15px;
}

.message-bubble {
  padding: 12px 16px;
  border-radius: 8px;
  background-color: #fff;
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);
  line-height: 1.6;
}

.message.user .message-bubble {
  background-color: #d9ecff;
}

.message-bubble :deep(p) {
  margin: 8px 0;
}

.message-bubble :deep(code) {
  background-color: #f5f7fa;
  padding: 2px 6px;
  border-radius: 4px;
}

.message-bubble :deep(pre) {
  background-color: #2d2d2d;
  color: #f8f8f2;
  padding: 12px;
  border-radius: 6px;
  overflow-x: auto;
}

.message-bubble :deep(pre code) {
  background: none;
  padding: 0;
}

.typing {
  display: flex;
  gap: 4px;
  padding: 20px 16px;
}

.typing .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #909399;
  animation: typing 1.4s infinite ease-in-out;
}

.typing .dot:nth-child(1) { animation-delay: -0.32s; }
.typing .dot:nth-child(2) { animation-delay: -0.16s; }

@keyframes typing {
  0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
  40% { transform: scale(1); opacity: 1; }
}

.streaming-status {
  margin-top: 6px;
  font-size: 12px;
  color: #409eff;
}

.thinking-block {
  margin-top: 8px;
  border: 1px solid #e4e7ed;
  border-radius: 6px;
  background: #fafafa;
  overflow: hidden;
}

.thinking-block summary {
  padding: 6px 10px;
  cursor: pointer;
  font-size: 12px;
  color: #606266;
  user-select: none;
  list-style: none;
}

.thinking-block summary::-webkit-details-marker {
  display: none;
}

.thinking-block summary::before {
  content: '▸';
  display: inline-block;
  margin-right: 6px;
  transition: transform 0.15s;
  color: #909399;
}

.thinking-block details[open] summary::before {
  transform: rotate(90deg);
}

.thinking-block .thinking-count {
  color: #c0c4cc;
  font-size: 11px;
  margin-left: 4px;
}

.thinking-block .thinking-content {
  padding: 4px 12px 10px;
  font-size: 12px;
  line-height: 1.7;
  color: #909399;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 260px;
  overflow-y: auto;
  border-top: 1px dashed #e4e7ed;
}

.process-time {
  margin-top: 8px;
  font-size: 12px;
  color: #909399;
  text-align: right;
}

.feedback-bar {
  margin-top: 8px;
  display: flex;
  gap: 6px;
}

.feedback-btn {
  background: none;
  border: 1px solid #e6e6e6;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 14px;
  cursor: pointer;
  line-height: 1.4;
  transition: all 0.15s;
}

.feedback-btn:hover {
  background-color: #f5f7fa;
  border-color: #d9d9d9;
}

.feedback-btn.active {
  background-color: #ecf5ff;
  border-color: #409eff;
}

.debug-info {
  margin-top: 10px;
  font-size: 12px;
}

.debug-info :deep(.el-collapse-item__header) {
  background-color: #f5f7fa;
  padding: 0 10px;
  border-radius: 4px;
  font-size: 12px;
  height: 32px;
  line-height: 32px;
}

.debug-toggle {
  font-weight: bold;
  color: #409eff;
}

.debug-section {
  margin-bottom: 12px;
  padding: 10px;
  background-color: #fafafa;
  border-radius: 4px;
}

.debug-title {
  font-weight: bold;
  color: #409eff;
  margin-bottom: 8px;
  font-size: 13px;
}

.debug-content {
  color: #606266;
  line-height: 1.8;
}

.debug-detail-section {
  margin: 12px 0;
  border: 1px solid #e6e6e6;
  border-radius: 4px;
  overflow: hidden;
}

.debug-subsection {
  border-bottom: 1px solid #e6e6e6;
}

.debug-subsection:last-child {
  border-bottom: none;
}

.debug-subtitle {
  background-color: #f0f7ff;
  padding: 8px 12px;
  cursor: pointer;
  font-weight: 500;
  color: #409eff;
  display: flex;
  align-items: center;
  gap: 6px;
}

.debug-subtitle:hover {
  background-color: #e6f0ff;
}

.toggle-icon {
  font-size: 10px;
  width: 12px;
  text-align: center;
}

.debug-detail-content {
  padding: 10px;
  background-color: #fff;
}

.type-results {
  margin-bottom: 12px;
}

.type-results:last-child {
  margin-bottom: 0;
}

.type-header {
  font-weight: bold;
  color: #67c23a;
  margin-bottom: 6px;
  padding: 4px 8px;
  background-color: #f0f9eb;
  border-radius: 3px;
  display: inline-block;
}

.result-item {
  padding: 8px;
  margin: 6px 0;
  background-color: #fafafa;
  border-radius: 4px;
  border-left: 3px solid #409eff;
}

.result-item.final {
  border-left-color: #67c23a;
  background-color: #f0f9eb;
}

.result-score {
  font-size: 11px;
  color: #909399;
  margin-right: 12px;
}

.result-scores {
  font-size: 11px;
  color: #909399;
  margin-bottom: 4px;
}

.result-scores span {
  margin-right: 12px;
}

.result-scores .highlight {
  color: #409eff;
  font-weight: bold;
}

.result-type-badge {
  display: inline-block;
  background-color: #409eff;
  color: white;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  margin-right: 8px;
}

.result-content {
  color: #606266;
  font-size: 11px;
  line-height: 1.5;
  display: block;
  margin-top: 4px;
}

.debug-note {
  color: #909399;
  font-style: italic;
}

.debug-step {
  padding: 4px 0;
  border-bottom: 1px dashed #e6e6e6;
}

.debug-step:last-child {
  border-bottom: none;
}

.step-name {
  display: inline-block;
  background-color: #409eff;
  color: white;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 11px;
  margin-right: 8px;
}

.step-name-inline {
  display: inline-block;
  background-color: #409eff;
  color: white;
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 11px;
  margin-right: 8px;
}

.step-count-inline {
  color: #67c23a;
  font-weight: bold;
  margin-left: 8px;
}

.step-desc {
  margin-right: 8px;
}

.step-count {
  color: #67c23a;
  font-weight: bold;
}

.step-scores {
  color: #909399;
  font-size: 11px;
  margin-left: 8px;
}

.debug-stats {
  margin: 8px 0;
}

.type-badge {
  display: inline-block;
  background-color: #e6f7ff;
  color: #409eff;
  padding: 2px 8px;
  border-radius: 3px;
  margin-right: 6px;
  font-size: 11px;
}

.context-preview {
  margin-top: 8px;
}

/* 步骤展开样式 */
.step-item {
  margin-bottom: 4px;
}

.step-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background-color: #fff;
  border: 1px solid #e6e6e6;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.step-header:hover {
  background-color: #f5f7fa;
  border-color: #d9d9d9;
}

.step-icon {
  font-size: 14px;
}

.step-name {
  flex: 1;
  font-size: 13px;
  color: #606266;
}

.step-count {
  color: #67c23a;
  font-weight: bold;
  font-size: 12px;
}

.step-time {
  color: #909399;
  font-size: 11px;
  margin-left: 6px;
}

.step-toggle {
  font-size: 12px;
  color: #909399;
  transition: transform 0.2s;
}

.step-details {
  margin-left: 20px;
  margin-top: 4px;
  padding: 10px;
  background-color: #fafafa;
  border-radius: 4px;
  border-left: 3px solid #409eff;
}

.step-results {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.step-info {
  font-size: 13px;
  color: #606266;
}

.context-preview pre {
  background-color: #f5f7fa;
  padding: 10px;
  border-radius: 4px;
  max-height: 200px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 11px;
  line-height: 1.5;
  margin: 8px 0 0 0;
}

.retrieved-sources {
  margin-top: 10px;
  font-size: 12px;
}

.source-item {
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}

.source-item:last-child {
  border-bottom: none;
}

.source-score {
  color: #909399;
  margin-bottom: 4px;
}

.source-content {
  color: #606266;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.6;
}

.input-area {
  padding: 15px 20px;
  background-color: #fff;
  border-top: 1px solid #e6e6e6;
}

.input-options {
  margin-bottom: 10px;
}

.input-wrapper {
  display: flex;
  gap: 10px;
  align-items: flex-end;
}

.input-wrapper .el-textarea {
  flex: 1;
}

/* 迭代分组样式 */
.iteration-group {
  background-color: #fafafa;
  border: 1px solid #e6e6e6;
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 12px;
}

.iteration-group:last-child {
  margin-bottom: 0;
}

.iteration-group-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid #e6e6e6;
}

.iteration-badge {
  background-color: #409eff;
  color: white;
  padding: 4px 10px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 13px;
}

.iteration-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.sufficient-badge {
  padding: 3px 8px;
  border-radius: 4px;
  font-weight: 500;
}

.sufficient-badge.pass {
  background-color: #f0f9eb;
  color: #67c23a;
}

.sufficient-badge.fail {
  background-color: #fef0f0;
  color: #f56c6c;
}

.iteration-steps {
  display: block;
  margin-bottom: 10px;
}

.iteration-step {
  display: flex;
  align-items: center;
  gap: 6px;
  background-color: #fff;
  padding: 6px 10px;
  border-radius: 4px;
  border: 1px solid #e6e6e6;
}

.step-icon {
  font-size: 14px;
}

.step-name {
  font-size: 13px;
  color: #606266;
}

.step-count {
  font-size: 12px;
  color: #909399;
}

.iteration-details {
  background-color: #fff;
  padding: 10px;
  border-radius: 4px;
  border: 1px solid #e6e6e6;
}

.detail-row {
  display: flex;
  margin-bottom: 6px;
}

.detail-row:last-child {
  margin-bottom: 0;
}

.detail-label {
  font-weight: 600;
  color: #909399;
  min-width: 80px;
  font-size: 13px;
}

.detail-value {
  color: #606266;
  font-size: 13px;
  flex: 1;
}

/* 评估信息样式 */
.evaluation-section {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed #e6e6e6;
}

.section-title {
  font-weight: 600;
  font-size: 13px;
  color: #409eff;
  margin-bottom: 8px;
}

.evaluation-item {
  display: flex;
  margin-bottom: 6px;
}

.evaluation-item:last-child {
  margin-bottom: 0;
}

/* 大模型相关样式 */
.llm-section {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed #e6e6e6;
}

.llm-prompt {
  background-color: #f5f7fa;
  padding: 10px;
  border-radius: 4px;
  font-size: 12px;
  line-height: 1.6;
  color: #606266;
  overflow-x: auto;
  max-height: 200px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}

.llm-response {
  background-color: #f0f9eb;
  padding: 10px;
  border-radius: 4px;
  font-size: 13px;
  line-height: 1.6;
  color: #67c23a;
  max-height: 300px;
  overflow-y: auto;
}

/* Agentic RAG 样式 */
.query-history {
  margin-top: 10px;
  padding: 10px;
  background-color: #f0f9eb;
  border-radius: 4px;
}

.query-item {
  padding: 4px 0;
  padding-left: 10px;
  border-left: 2px solid #67c23a;
  margin: 4px 0;
  font-size: 12px;
}

.execution-log {
  margin-top: 12px;
}

.log-item {
  padding: 8px;
  margin: 6px 0;
  background-color: #fff;
  border-radius: 4px;
  border-left: 3px solid #409eff;
}

.log-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
}

.log-step {
  background-color: #409eff;
  color: white;
  padding: 2px 8px;
  border-radius: 3px;
  font-size: 11px;
}

.log-time {
  color: #909399;
  font-size: 11px;
}

.log-details {
  font-size: 11px;
  color: #606266;
}

.log-details span {
  display: inline-block;
  margin-right: 12px;
}

/* 阶段用时样式 */
.timing-info {
  margin-top: 12px;
}

.timing-item {
  padding: 10px;
  margin: 8px 0;
  background-color: #fff;
  border-radius: 4px;
  border-left: 3px solid #e6a23c;
}

.timing-header {
  font-weight: bold;
  color: #303133;
  margin-bottom: 8px;
  font-size: 12px;
}

.timing-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.timing-step {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background-color: #f5f7fa;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 11px;
}

.timing-step span:first-child {
  color: #606266;
}

.timing-value {
  color: #e6a23c;
  font-weight: bold;
}

.timing-total {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed #e6e6e6;
  font-size: 11px;
  color: #909399;
}

.generation-timing {
  margin-top: 12px;
  padding: 10px;
  background-color: #f0f9eb;
  border-radius: 4px;
  border-left: 3px solid #67c23a;
}

.generation-timing .timing-step {
  background-color: #e1f3d8;
}
</style>

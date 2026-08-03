<template>
  <div class="admin-documents-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>所有文档管理</span>
        </div>
      </template>

      <el-table :data="documents" v-loading="loading">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="title" label="标题" />
        <el-table-column prop="file_name" label="文件名" width="180" />
        <el-table-column label="上传用户" width="120">
          <template #default="{ row }">
            {{ row.user?.username || '-' }}
          </template>
        </el-table-column>
        <el-table-column label="状态" width="100" align="center">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">{{ getStatusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="上传时间" width="180">
          <template #default="{ row }">
            {{ formatDate(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="100" align="center">
          <template #default="{ row }">
            <el-button type="danger" text @click="handleDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getAllDocuments, adminDeleteDocument } from '@/api/admin'

const documents = ref([])
const loading = ref(false)

onMounted(() => {
  fetchDocuments()
})

async function fetchDocuments() {
  loading.value = true
  try {
    documents.value = await getAllDocuments()
  } finally {
    loading.value = false
  }
}

async function handleDelete(row) {
  try {
    await ElMessageBox.confirm(`确定要删除文档「${row.title}」吗？`, '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await adminDeleteDocument(row.id)
    ElMessage.success('删除成功')
    fetchDocuments()
  } catch {
  }
}

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleString('zh-CN')
}

function getStatusType(status) {
  const types = {
    pending: 'info',
    processing: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return types[status] || 'info'
}

function getStatusText(status) {
  const texts = {
    pending: '待处理',
    processing: '处理中',
    completed: '已完成',
    failed: '失败'
  }
  return texts[status] || status
}
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>

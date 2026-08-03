<template>
  <div class="documents-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>我的文档</span>
          <el-button type="primary" @click="showUploadDialog = true">
            <el-icon><Upload /></el-icon>
            上传文档
          </el-button>
        </div>
      </template>

      <el-table :data="documents" v-loading="loading">
        <el-table-column prop="title" label="标题" />
        <el-table-column prop="file_name" label="文件名" width="200" />
        <el-table-column prop="file_type" label="类型" width="80" align="center" />
        <el-table-column label="大小" width="100" align="center">
          <template #default="{ row }">
            {{ formatFileSize(row.file_size) }}
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
        <el-table-column label="操作" width="150" align="center">
          <template #default="{ row }">
            <el-button type="danger" text @click="handleDelete(row)">
              <el-icon><Delete /></el-icon>
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="showUploadDialog" title="上传文档" width="500px">
      <el-form :model="uploadForm" label-width="80px">
        <el-form-item label="标题">
          <el-input v-model="uploadForm.title" placeholder="可选，默认为文件名" />
        </el-form-item>
        <el-form-item label="文件">
          <el-upload
            ref="uploadRef"
            :auto-upload="false"
            :on-change="handleFileChange"
            :limit="1"
            accept=".pdf,.txt,.docx,.md"
          >
            <el-button type="primary">选择文件</el-button>
            <template #tip>
              <div class="el-upload__tip">
                支持 PDF, TXT, DOCX, MD 格式，最大 50MB
              </div>
            </template>
          </el-upload>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showUploadDialog = false">取消</el-button>
        <el-button type="primary" @click="handleUpload" :loading="uploading">上传</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Upload, Delete } from '@element-plus/icons-vue'
import { getDocuments, uploadDocument, deleteDocument } from '@/api/document'

const documents = ref([])
const loading = ref(false)
const showUploadDialog = ref(false)
const uploading = ref(false)
const uploadRef = ref(null)
const selectedFile = ref(null)

const uploadForm = reactive({
  title: ''
})

onMounted(() => {
  fetchDocuments()
})

async function fetchDocuments() {
  loading.value = true
  try {
    documents.value = await getDocuments()
  } finally {
    loading.value = false
  }
}

function handleFileChange(file) {
  selectedFile.value = file.raw
}

async function handleUpload() {
  if (!selectedFile.value) {
    ElMessage.warning('请选择文件')
    return
  }

  uploading.value = true
  try {
    await uploadDocument(selectedFile.value, uploadForm.title || null)
    ElMessage.success('文档上传成功，正在处理中...')
    showUploadDialog.value = false
    uploadForm.title = ''
    selectedFile.value = null
    if (uploadRef.value) {
      uploadRef.value.clearFiles()
    }
    fetchDocuments()
  } finally {
    uploading.value = false
  }
}

async function handleDelete(row) {
  try {
    await ElMessageBox.confirm(`确定要删除文档「${row.title}」吗？`, '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await deleteDocument(row.id)
    ElMessage.success('删除成功')
    fetchDocuments()
  } catch {
  }
}

function formatFileSize(bytes) {
  if (!bytes) return '-'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
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

<template>
  <div class="admin-users-container">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>用户管理</span>
        </div>
      </template>

      <el-table :data="users" v-loading="loading">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="username" label="用户名" width="150" />
        <el-table-column prop="email" label="邮箱" width="200" />
        <el-table-column label="角色" width="100" align="center">
          <template #default="{ row }">
            <el-tag :type="row.role === 'admin' ? 'danger' : ''">
              {{ row.role === 'admin' ? '管理员' : '用户' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="80" align="center">
          <template #default="{ row }">
            <el-tag :type="row.is_active ? 'success' : 'info'" size="small">
              {{ row.is_active ? '启用' : '禁用' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="注册时间" width="180">
          <template #default="{ row }">
            {{ formatDate(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" align="center">
          <template #default="{ row }">
            <el-button type="primary" text @click="handleEditRole(row)">修改角色</el-button>
            <el-button type="danger" text @click="handleDelete(row)" :disabled="row.is_current">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="showEditDialog" title="修改用户角色" width="400px">
      <el-form :model="editForm" label-width="80px">
        <el-form-item label="用户名">
          <el-input :value="currentEditUser?.username" disabled />
        </el-form-item>
        <el-form-item label="角色">
          <el-select v-model="editForm.role">
            <el-option label="用户" value="user" />
            <el-option label="管理员" value="admin" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showEditDialog = false">取消</el-button>
        <el-button type="primary" @click="handleSaveRole" :loading="saving">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getUsers, updateUser, deleteUser } from '@/api/admin'
import { useUserStore } from '@/store/user'

const userStore = useUserStore()
const users = ref([])
const loading = ref(false)
const showEditDialog = ref(false)
const saving = ref(false)
const currentEditUser = ref(null)

const editForm = reactive({
  role: 'user'
})

onMounted(() => {
  fetchUsers()
})

const usersWithCurrent = computed(() => {
  return users.value.map(u => ({
    ...u,
    is_current: u.id === userStore.user?.id
  }))
})

async function fetchUsers() {
  loading.value = true
  try {
    users.value = await getUsers()
  } finally {
    loading.value = false
  }
}

function handleEditRole(user) {
  currentEditUser.value = user
  editForm.role = user.role
  showEditDialog.value = true
}

async function handleSaveRole() {
  if (!currentEditUser.value) return

  saving.value = true
  try {
    await updateUser(currentEditUser.value.id, { role: editForm.role })
    ElMessage.success('修改成功')
    showEditDialog.value = false
    fetchUsers()
  } finally {
    saving.value = false
  }
}

async function handleDelete(user) {
  try {
    await ElMessageBox.confirm(`确定要删除用户「${user.username}」吗？`, '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await deleteUser(user.id)
    ElMessage.success('删除成功')
    fetchUsers()
  } catch {
  }
}

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleString('zh-CN')
}
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>

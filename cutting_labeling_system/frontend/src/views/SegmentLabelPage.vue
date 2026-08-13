<template>
  <div class="page-container">
    <div class="header">
      <a-button @click="goBack" style="margin-right: 16px;">← 返回主页</a-button>
      <h1 class="title">标注数据展示</h1>
      <a-tag v-if="currentProject" color="blue">{{ currentProject }}</a-tag>
    </div>

    <div class="stats-panel">
      <div class="stat-card">
        <a-statistic title="总样本数" :value="stats.total_samples" />
      </div>
      <div class="stat-card">
        <a-statistic title="已标注" :value="stats.annotated_count" />
      </div>
      <div class="stat-card">
        <a-statistic title="待标注" :value="stats.unannotated_count" />
      </div>
    </div>

    <div style="margin-bottom: 16px; display: flex; gap: 12px; flex-wrap: wrap; align-items: center;">
      <a-input
        v-model:value="searchKeyword"
        placeholder="搜索 line_id"
        style="width: 350px"
        @keyup.enter="handleSearch"
        allow-clear
      >
        <template #prefix>
          <SearchOutlined />
        </template>
      </a-input>
      <a-button type="primary" @click="handleSearch">搜索</a-button>
      <a-button @click="clearSearch" v-if="searchKeyword">清除</a-button>

      <a-select
        v-model:value="isAnnotatedFilter"
        placeholder="标注状态"
        style="width: 160px"
        @change="loadList"
      >
        <a-select-option :value="null">全部</a-select-option>
        <a-select-option :value="true">已标注</a-select-option>
        <a-select-option :value="false">待标注</a-select-option>
      </a-select>

      <a-button type="primary" @click="goNextLabel">快速标注</a-button>
    </div>

    <div class="image-grid">
      <div
        v-for="(item, index) in allData"
        :key="item.id || `item_${index}`"
        class="image-card"
        :class="{ 'annotated': item.is_annotated }"
        @click="goLabel(item.id)"
      >
        <div class="card-img">
          <img :src="getImageUrl(item.id)" :alt="item.line_id" />
        </div>
        <div class="card-info">
          <div class="card-title">{{ item.line_id }}</div>
          <div class="card-meta">
            <a-tag :color="item.is_annotated ? 'green' : (item.is_postponed ? 'orange' : 'gray')">
              {{ item.is_annotated ? '已标注' : (item.is_postponed ? '暂不标注' : '待标注') }}
            </a-tag>
            <span class="char-count">字符数: {{ item.char_count }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="pagination-container">
      <a-pagination
        v-model:current="pagination.current"
        v-model:page-size="pagination.pageSize"
        :total="pagination.total"
        :page-size-options="[10, 20, 50, 100]"
        show-size-changer
        :show-total="(total) => `共 ${total} 条`"
        @change="handlePageChange"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Button as aButton, Statistic as aStatistic, Pagination as aPagination, Input as aInput, Tag as aTag, Select as aSelect, SelectOption as aSelectOption } from 'ant-design-vue'
import { SearchOutlined } from '@ant-design/icons-vue'
import { imagesApi, annotationApi } from '../api'

const router = useRouter()
const route = useRoute()

const currentProject = ref(null)
const allData = ref([])
const searchKeyword = ref('')
const isSearching = ref(false)
const isAnnotatedFilter = ref(null)
const stats = ref({
  total_samples: 0,
  annotated_count: 0,
  unannotated_count: 0,
  postponed_count: 0
})

const pagination = ref({
  pageSize: 20,
  current: 1,
  total: 0
})

const getImageUrl = (lineId) => {
  return imagesApi.raw(lineId, currentProject.value)
}

const handleSearch = async () => {
  if (!searchKeyword.value.trim()) {
    clearSearch()
    return
  }
  isSearching.value = true
  pagination.value.current = 1
  const filtered = allData.value.filter(item => 
    item.line_id.toLowerCase().includes(searchKeyword.value.toLowerCase())
  )
  allData.value = filtered
  pagination.value.total = filtered.length
}

const clearSearch = () => {
  searchKeyword.value = ''
  isSearching.value = false
  pagination.value.current = 1
  loadList()
}

const handlePageChange = (page, pageSize) => {
  pagination.value.current = page
  pagination.value.pageSize = pageSize
  loadList()
}

const loadList = async () => {
  if (isSearching.value) {
    return
  }
  try {
    const sendParams = {
      page: pagination.value.current,
      page_size: pagination.value.pageSize,
      is_annotated: isAnnotatedFilter.value
    }
    if (currentProject.value) {
      sendParams.project_id = currentProject.value
    }
    const res = await imagesApi.list(sendParams)
    allData.value = Array.isArray(res.data?.data) ? res.data.data : []
    pagination.value.total = Number(res.data?.total) || 0
  } catch (err) {
    allData.value = []
    pagination.value.total = 0
    console.error("请求失败：", err)
  }
}

const loadStats = async () => {
  try {
    const res = await annotationApi.stats(currentProject.value)
    if (res.data?.code === 0) {
      stats.value = res.data.data
    }
  } catch (err) {
    stats.value = {
      total_samples: 0,
      annotated_count: 0,
      unannotated_count: 0,
      postponed_count: 0
    }
    console.error("加载统计数据失败：", err)
  }
}

const goLabel = (id) => {
  const url = `/label/${id}?project_id=${currentProject.value}`
  window.open(url, '_blank')
}

const goBack = () => {
  router.push('/')
}

const goNextLabel = async () => {
  try {
    const res = await annotationApi.next(currentProject.value)
    if (res.data?.code === 0 && res.data.data) {
      router.push({ path: `/label/${res.data.data.line_id}`, query: { project_id: currentProject.value } })
    } else {
      alert('没有更多待标注样本')
    }
  } catch (err) {
    console.error('获取下一个标注失败：', err)
  }
}

const initData = () => {
  currentProject.value = route.query.project_id || null
  loadList()
  loadStats()
}

const handleStorageChange = (e) => {
  if (e.key === 'annotation_status_change') {
    try {
      const statusData = JSON.parse(e.newValue)
      if (statusData.project_id === currentProject.value) {
        loadList()
        loadStats()
      }
    } catch (error) {
      console.error('解析状态变更数据失败:', error)
    }
  }
}

onMounted(() => {
  initData()
  window.addEventListener('storage', handleStorageChange)
})

onUnmounted(() => {
  window.removeEventListener('storage', handleStorageChange)
})

watch(() => route.fullPath, () => {
  initData()
})
</script>

<style scoped>
.page-container { padding: 30px; max-width: 1400px; margin: 0 auto; }
.header { margin-bottom: 24px; display: flex; align-items: center; }
.title { margin: 0 16px 0 0; font-size: 28px; }
.stats-panel { display: flex; gap: 16px; margin-bottom: 20px; padding: 20px; background: #f5f5f5; border-radius: 8px; flex-wrap: wrap; }
.stat-card { flex: 1; min-width: 150px; background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; margin-bottom: 30px; }
.image-card { background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); cursor: pointer; transition: all 0.3s ease; border: 2px solid #e8e8e8; }
.image-card:hover { border-color: #1890ff; box-shadow: 0 4px 16px rgba(24,144,255,0.15); transform: translateY(-4px); }
.image-card.annotated { border-color: #52c41a; background: #f6ffed; }
.image-card.annotated:hover { border-color: #52c41a; box-shadow: 0 4px 16px rgba(82,196,26,0.2); }
.card-img { padding: 12px; background: #f8f8f8; display: flex; align-items: center; justify-content: center; }
.card-img img { max-width: 100%; max-height: 120px; object-fit: contain; }
.card-info { padding: 12px 16px; }
.card-title { font-size: 14px; font-weight: 500; color: #333; margin-bottom: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.card-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.char-count { font-size: 12px; color: #666; }
.pagination-container { display: flex; justify-content: center; }

:deep(.ant-pagination-options) {
  .ant-select {
    width: 100px;
  }
  .ant-select-dropdown {
    min-width: 100px !important;
    width: auto !important;
  }
}
</style>
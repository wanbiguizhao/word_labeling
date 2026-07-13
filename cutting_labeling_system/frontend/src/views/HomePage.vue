<template>
  <div class="home-container">
    <div class="header">
      <h1 class="title">汉字切割标注系统</h1>
      <p class="subtitle">多项目标注管理平台</p>
    </div>

    <div class="project-section">
      <div class="section-header">
        <span class="section-icon">📁</span>
        <h2 class="section-title">项目管理</h2>
        <a-button type="primary" @click="showCreateModal = true">创建项目</a-button>
      </div>

      <div class="project-selector">
        <a-select
          v-model:value="currentProject"
          placeholder="选择项目"
          style="width: 350px"
          @change="handleProjectChange"
        >
          <a-select-option v-for="p in projects" :key="p.project_id" :value="p.project_id">
            {{ p.project_name }} ({{ p.line_id_count }}条)
          </a-select-option>
        </a-select>
        <a-tag v-if="currentProjectName" color="blue">{{ currentProjectName }}</a-tag>
      </div>

      <div v-if="projects.length > 0" class="project-list">
        <a-card
          v-for="project in projects"
          :key="project.project_id"
          :class="{ 'active': project.project_id === currentProject }"
          @click="selectProject(project.project_id)"
        >
          <div class="project-card-header">
            <h3>{{ project.project_name }}</h3>
            <a-popconfirm title="确定删除项目？" @confirm="handleDeleteProject(project.project_id)">
              <a-button size="small" danger v-if="project.project_id !== currentProject">删除</a-button>
            </a-popconfirm>
          </div>
          <div class="project-card-info">
            <div class="info-row">
              <span class="info-label">描述:</span>
              <span class="info-value">{{ project.description || '无' }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">选择策略:</span>
              <span class="info-value">{{ getStrategyName(project.selector?.strategy) }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">PDF文件:</span>
              <span class="info-value">{{ project.pdf_count }}个</span>
            </div>
            <div class="info-row">
              <span class="info-label">数据量:</span>
              <span class="info-value">{{ project.line_id_count }}条</span>
            </div>
          </div>
          <div class="project-card-stats">
            <a-progress
              :percent="project.line_id_count > 0 ? Math.round((project.annotated_count / project.line_id_count) * 100) : 0"
              :show-info="false"
              stroke-color="#52c41a"
              style="flex: 1; margin-right: 16px;"
            />
            <span class="stats-text">{{ project.annotated_count }}/{{ project.line_id_count }}</span>
          </div>
        </a-card>
      </div>
    </div>

    <div class="stats-section">
      <div class="section-header">
        <span class="section-icon">📊</span>
        <h2 class="section-title">标注统计</h2>
      </div>
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon">📝</div>
          <div class="stat-info">
            <a-statistic title="总样本数" :value="stats.total_samples" />
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">✅</div>
          <div class="stat-info">
            <a-statistic title="已标注" :value="stats.annotated_count" />
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">⏳</div>
          <div class="stat-info">
            <a-statistic title="暂不标注" :value="stats.postponed_count" />
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">📋</div>
          <div class="stat-info">
            <a-statistic title="待标注" :value="stats.unannotated_count" />
          </div>
        </div>
      </div>
    </div>

    <div class="category-section">
      <div class="section-header">
        <span class="section-icon">🎯</span>
        <h2 class="section-title">功能入口</h2>
      </div>
      <div class="category-grid">
        <div class="feature-card" @click="goTo('/segment')">
          <div class="feature-icon">
            <span class="icon">🖼️</span>
          </div>
          <h3 class="feature-title">标注图片列表</h3>
          <p class="feature-desc">浏览所有标注图片，支持筛选和排序</p>
          <div class="feature-tags">
            <a-tag color="blue">图片浏览</a-tag>
            <a-tag color="green">标注查看</a-tag>
          </div>
        </div>
        <div class="feature-card" @click="goNextLabel">
          <div class="feature-icon">
            <span class="icon">➡️</span>
          </div>
          <h3 class="feature-title">快速标注</h3>
          <p class="feature-desc">直接跳转到下一个待标注样本</p>
          <div class="feature-tags">
            <a-tag color="orange">智能跳转</a-tag>
            <a-tag color="green">高效标注</a-tag>
          </div>
        </div>
      </div>
    </div>

    <div class="footer">
      <p>规则存储路径：datahome/rule_jsons</p>
    </div>

    <a-modal
      v-model:open="showCreateModal"
      title="创建项目"
      ok-text="创建"
      cancel-text="取消"
      @ok="handleCreateProject"
    >
      <a-form :model="createForm" :label-col="{ span: 6 }" :wrapper-col="{ span: 18 }">
        <a-form-item label="项目名称">
          <a-input v-model:value="createForm.project_name" placeholder="请输入项目名称" />
        </a-form-item>
        <a-form-item label="项目描述">
          <a-textarea v-model:value="createForm.description" placeholder="请输入项目描述" :rows="3" />
        </a-form-item>
        <a-form-item label="选择策略">
          <a-select v-model:value="createForm.strategy" placeholder="选择策略">
            <a-select-option v-for="s in strategies" :key="s.strategy" :value="s.strategy">
              {{ s.name }} - {{ s.description }}
            </a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="PDF文件ID" v-if="createForm.strategy === 'pdf'">
          <a-input v-model:value="createForm.pdf_ids" placeholder="多个ID用逗号分隔，如 gwyb195401,gwyb195402" />
        </a-form-item>
        <a-form-item label="数据量">
          <a-input-number v-model:value="createForm.count" :min="1" :max="10000" placeholder="选择数据条数" />
        </a-form-item>
        <a-form-item>
          <a-button type="dashed" @click="previewSelector" style="width: 100%">预览选择结果</a-button>
          <div v-if="previewResult" class="preview-result">
            <p>策略: {{ previewResult.strategy }}</p>
            <p>未标注总数: {{ previewResult.total_unannotated }}</p>
            <p>选中数量: {{ previewResult.selected_count }}</p>
            <p>示例ID: {{ previewResult.sample_line_ids?.join(', ') }}</p>
          </div>
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { 
  Select as aSelect, 
  SelectOption as aSelectOption, 
  Tag as aTag, 
  Statistic as aStatistic,
  Button as aButton,
  Card as aCard,
  Modal as aModal,
  Form as aForm,
  FormItem as aFormItem,
  Input as aInput,
  Textarea as aTextarea,
  InputNumber as aInputNumber,
  Progress as aProgress,
  Popconfirm as aPopconfirm
} from 'ant-design-vue'
import { projectsApi, annotationApi, selectorsApi } from '../api'

const router = useRouter()

const projects = ref([])
const currentProject = ref(null)
const stats = ref({
  total_samples: 0,
  annotated_count: 0,
  postponed_count: 0,
  unannotated_count: 0
})

const showCreateModal = ref(false)
const strategies = ref([])
const previewResult = ref(null)
const createForm = ref({
  project_name: '',
  description: '',
  strategy: 'random',
  pdf_ids: '',
  count: 100
})

const currentProjectName = computed(() => {
  const project = projects.value.find(p => p.project_id === currentProject.value)
  return project?.project_name || ''
})

const getStrategyName = (strategy) => {
  const s = strategies.value.find(item => item.strategy === strategy)
  return s?.name || strategy
}

const loadProjects = async () => {
  try {
    const res = await projectsApi.list()
    if (res.data?.code === 0) {
      projects.value = res.data.data.projects || []
      currentProject.value = res.data.data.current_project || null
    }
  } catch (err) {
    console.error('加载项目失败：', err)
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
    console.error('加载统计数据失败：', err)
  }
}

const loadStrategies = async () => {
  try {
    const res = await selectorsApi.strategies()
    if (res.data?.code === 0) {
      strategies.value = res.data.data || []
    }
  } catch (err) {
    strategies.value = [
      { strategy: 'random', name: '随机选择', description: '从所有未标注数据中随机选择' }
    ]
  }
}

const handleProjectChange = async (projectId) => {
  try {
    await projectsApi.switch(projectId)
  } catch (err) {
    console.error('切换项目失败：', err)
  }
}

// 当 currentProject 变化时自动重新加载统计数据（比 @change 更可靠）
watch(currentProject, (newVal) => {
  if (newVal) {
    loadStats()
  }
})

const selectProject = (projectId) => {
  currentProject.value = projectId
  handleProjectChange(projectId)
}

const handleCreateProject = async () => {
  if (!createForm.value.project_name) {
    alert('请输入项目名称')
    return
  }
  
  try {
    const data = {
      project_name: createForm.value.project_name,
      description: createForm.value.description,
      strategy: createForm.value.strategy,
      count: createForm.value.count
    }
    
    if (createForm.value.strategy === 'pdf') {
      data.pdf_ids = createForm.value.pdf_ids.split(',').map(id => id.trim()).filter(id => id)
      if (data.pdf_ids.length === 0) {
        alert('PDF选择器需要指定PDF文件ID')
        return
      }
    }
    
    await projectsApi.create(data)
    showCreateModal.value = false
    createForm.value = {
      project_name: '',
      description: '',
      strategy: 'random',
      pdf_ids: '',
      count: 100
    }
    previewResult.value = null
    await loadProjects()
    alert('项目创建成功')
  } catch (err) {
    console.error('创建项目失败：', err)
    alert('创建项目失败')
  }
}

const handleDeleteProject = async (projectId) => {
  try {
    await projectsApi.delete(projectId)
    await loadProjects()
  } catch (err) {
    console.error('删除项目失败：', err)
    alert('删除项目失败')
  }
}

const previewSelector = async () => {
  try {
    const data = {
      strategy: createForm.value.strategy,
      count: createForm.value.count
    }
    
    if (createForm.value.strategy === 'pdf') {
      data.pdf_ids = createForm.value.pdf_ids.split(',').map(id => id.trim()).filter(id => id)
    }
    
    const res = await selectorsApi.preview(data)
    if (res.data?.code === 0) {
      previewResult.value = res.data.data
    }
  } catch (err) {
    console.error('预览选择失败：', err)
  }
}

const goTo = (path) => {
  router.push({ path, query: { project_id: currentProject.value } })
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

onMounted(async () => {
  await loadProjects()
  loadStrategies()
  // loadStats 由 watch(currentProject) 自动触发
})
</script>

<style scoped>
.home-container { padding: 40px; max-width: 1400px; margin: 0 auto; }
.header { text-align: center; margin-bottom: 40px; }
.title { font-size: 36px; font-weight: 600; color: #1a1a1a; margin-bottom: 12px; }
.subtitle { font-size: 18px; color: #666; }
.project-section { margin-bottom: 40px; }
.section-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 2px solid #e8e8e8; }
.section-icon { font-size: 28px; }
.section-title { font-size: 24px; font-weight: 600; color: #1a1a1a; margin: 0; flex: 1; }
.project-selector { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
.project-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
.project-card { cursor: pointer; transition: all 0.3s ease; }
.project-card.active { border-color: #1890ff; box-shadow: 0 4px 16px rgba(24,144,255,0.15); }
.project-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.project-card-header h3 { margin: 0; font-size: 18px; }
.project-card-info { margin-bottom: 16px; }
.info-row { display: flex; margin-bottom: 8px; }
.info-label { color: #999; width: 80px; flex-shrink: 0; }
.info-value { color: #333; flex: 1; }
.project-card-stats { display: flex; align-items: center; padding-top: 12px; border-top: 1px solid #f0f0f0; }
.stats-text { font-size: 14px; color: #666; min-width: 80px; text-align: right; }
.stats-section { margin-bottom: 40px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 20px; }
.stat-card { background: #fff; border-radius: 12px; padding: 24px; display: flex; align-items: center; gap: 16px; border: 2px solid #e8e8e8; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.stat-icon { font-size: 36px; }
.stat-info { flex: 1; }
.category-section { margin-bottom: 40px; }
.category-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }
.feature-card { background: #fff; border-radius: 12px; padding: 28px; cursor: pointer; transition: all 0.3s ease; border: 2px solid #e8e8e8; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.feature-card:hover { border-color: #1890ff; box-shadow: 0 4px 16px rgba(24,144,255,0.15); transform: translateY(-4px); }
.feature-icon { margin-bottom: 14px; }
.icon { font-size: 40px; }
.feature-title { font-size: 20px; font-weight: 600; color: #1a1a1a; margin-bottom: 10px; }
.feature-desc { font-size: 14px; color: #666; line-height: 1.6; margin-bottom: 14px; }
.feature-tags { display: flex; gap: 8px; }
.footer { text-align: center; color: #999; font-size: 14px; padding-top: 20px; border-top: 1px solid #e8e8e8; margin-top: 20px; }
.preview-result { margin-top: 16px; padding: 12px; background: #f5f5f5; border-radius: 8px; }
.preview-result p { margin: 4px 0; font-size: 14px; }
</style>
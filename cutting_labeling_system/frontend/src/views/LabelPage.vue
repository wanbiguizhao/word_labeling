<template>
  <div class="label-page">
    <div class="header">
      <a-button @click="goBack" size="large">返回列表</a-button>
      <h2>{{ lineId }}</h2>
      <a-tag v-if="currentProject" color="blue">{{ currentProject }}</a-tag>
      <span class="char-count">字符数: {{ charCount }}</span>
    </div>

    <div v-if="loading" class="loading-container">
      <a-spin tip="加载中..." size="large" />
    </div>

    <div v-else class="content">
      <div class="section-card">
        <div class="section-header">
          <h3>区域一：原始行图片展示</h3>
        </div>
        <div class="section-body">
          <img :src="imgUrl" class="original-image" />
        </div>
      </div>

      <div class="section-card">
        <div class="section-header">
          <h3>区域二：规则切割结果预览（只读）</h3>
          <div class="header-actions">
            <span class="line-count">切割线数: {{ ruleLines.length }}</span>
            <a-button 
              type="dashed" 
              size="small" 
              @click="copySelectedLines('rule')"
              :disabled="ruleSelectedIndexes.length === 0"
            >
              复制选中分割线
            </a-button>
            <a-button 
              type="primary" 
              size="small" 
              @click="useScheme('rule')"
            >
              使用规则切割
            </a-button>
          </div>
        </div>
        <div class="section-body">
          <LineCanvas
            :image-url="imgUrl"
            :lines="ruleLines"
            :selected-indexes="ruleSelectedIndexes"
            :readonly="true"
            @update:selected-indexes="(idxs) => ruleSelectedIndexes = idxs"
          />
        </div>
      </div>

      <div class="section-card">
        <div class="section-header">
          <h3>区域三：模型切割结果预览（只读）</h3>
          <div class="header-actions">
            <span class="line-count">切割线数: {{ modelLines.length }}</span>
            <a-button 
              type="dashed" 
              size="small" 
              @click="copySelectedLines('model')"
              :disabled="modelSelectedIndexes.length === 0"
            >
              复制选中分割线
            </a-button>
            <a-button 
              type="primary" 
              size="small" 
              @click="useScheme('model')"
            >
              使用模型切割
            </a-button>
          </div>
        </div>
        <div class="section-body">
          <LineCanvas
            :image-url="imgUrl"
            :lines="modelLines"
            :selected-indexes="modelSelectedIndexes"
            :readonly="true"
            @update:selected-indexes="(idxs) => modelSelectedIndexes = idxs"
          />
        </div>
      </div>

      <div class="section-card">
        <div class="section-header">
          <h3>区域四：融合切割结果预览（只读）</h3>
          <div class="header-actions">
            <span class="line-count">切割线数: {{ fusionLines.length }}</span>
            <a-button 
              type="dashed" 
              size="small" 
              @click="copySelectedLines('fusion')"
              :disabled="fusionSelectedIndexes.length === 0"
            >
              复制选中分割线
            </a-button>
            <a-button 
              type="primary" 
              size="small" 
              @click="useScheme('fusion')"
            >
              使用融合切割
            </a-button>
          </div>
        </div>
        <div class="section-body">
          <LineCanvas
            :image-url="imgUrl"
            :lines="fusionLines"
            :selected-indexes="fusionSelectedIndexes"
            :readonly="true"
            @update:selected-indexes="(idxs) => fusionSelectedIndexes = idxs"
          />
        </div>
      </div>

      <div class="section-card editable-card">
        <div class="section-header">
          <h3>区域五：可编辑切割线区域</h3>
          <div v-if="isEditing" class="line-color-selector">
            <span class="color-label">新增线色：</span>
            <a-radio-group v-model:value="currentLineColor" size="small">
              <a-radio-button :value="null">
                无
              </a-radio-button>
              <a-radio-button value="#ff0000">
                <span style="color: #ff0000;">■</span> 红线(开始)
              </a-radio-button>
              <a-radio-button value="#9932cc">
                <span style="color: #9932cc;">■</span> 紫线(共享)
              </a-radio-button>
              <a-radio-button value="#00ff00">
                <span style="color: #00aa00;">■</span> 绿线(结束)
              </a-radio-button>
            </a-radio-group>
          </div>
          <div v-if="isEditing" class="zoom-controls">
            <span class="zoom-label">缩放：</span>
            <a-button size="small" @click="zoomOut">−</a-button>
            <span class="zoom-value">{{ Math.round(zoomLevel * 100) }}%</span>
            <a-button size="small" @click="zoomIn">+</a-button>
            <a-button size="small" @click="resetZoom">100%</a-button>
          </div>
        </div>
        <div class="section-body">
          <div class="edit-hint" v-if="!isEditing">
            点击下方"开始编辑"或使用"使用规则/模型/融合切割"按钮进入编辑模式
          </div>
          <div class="edit-hint" v-else>
            <strong>编辑模式：</strong>
            <span>单击选择（8px范围内）| Ctrl+拖拽框选多选 | Delete删除 | 方向键微调位置 | 点击空白处添加线条</span>
            <br>
            <strong>线色说明：</strong>
            <span><span style="color: #ff0000;">■</span>红线=字符开始 | <span style="color: #9932cc;">■</span>紫线=共享边界（上一字结束=下一字开始） | <span style="color: #00aa00;">■</span>绿线=字符结束</span>
          </div>
          <div class="editable-canvas-wrapper">
            <LineCanvas
              :image-url="imgUrl"
              :lines="currentEditableLines"
              :selected-indexes="selectedIndexes"
              :readonly="!isEditing"
              :line-color="currentLineColor"
              :zoom="zoomLevel"
              @update:lines="handleLinesChange"
              @update:selected-indexes="handleSelectedIndexesChange"
            />
          </div>
        </div>
      </div>

      <div class="action-bar">
        <div class="action-section">
          <span class="section-label">编辑操作</span>
          <a-button 
            v-if="!isEditing" 
            type="primary" 
            @click="startEditing"
          >
            开始编辑
          </a-button>
          <a-button 
            v-if="isEditing" 
            @click="deleteSelected"
            :disabled="selectedIndexes.length === 0"
          >
            删除选中
          </a-button>
          <a-button 
            v-if="isEditing" 
            @click="clearAll"
            :disabled="editedLines.length === 0"
          >
            清空所有
          </a-button>
          <a-button v-if="isEditing" @click="cancelEditing">取消编辑</a-button>
        </div>

        <div class="action-section">
          <span class="section-label">保存操作</span>
          <a-button 
            v-if="isEditing" 
            type="primary" 
            @click="saveAnnotation"
          >
            保存
          </a-button>
          <a-button 
            v-if="!is_annotated && !is_postponed" 
            @click="handlePostpone"
          >
            暂不标注
          </a-button>
          <a-button 
            v-if="is_postponed" 
            type="dashed" 
            @click="handleUnpostpone"
          >
            取消暂不标注
          </a-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Button as aButton, Tag as aTag, Spin as aSpin, Table as aTable, TableColumn as aTableColumn, RadioGroup as aRadioGroup, RadioButton as aRadioButton } from 'ant-design-vue'
import LineCanvas from '../components/LineCanvas.vue'
import { imagesApi } from '../api'

const route = useRoute()
const router = useRouter()

const lineId = route.params.id
const currentProject = ref(route.query.project_id || null)

const imgUrl = computed(() => imagesApi.raw(lineId, currentProject.value))

const loading = ref(true)
const charCount = ref(0)
const chars = ref([])
const ruleLines = ref([])
const modelLines = ref([])
const fusionLines = ref([])
const annotationLines = ref([])
const is_annotated = ref(false)
const is_postponed = ref(false)

const isEditing = ref(false)
const editedLines = ref([])
const selectedIndexes = ref([])
const currentLineColor = ref(null)
const zoomLevel = ref(1.0)

const ruleSelectedIndexes = ref([])
const modelSelectedIndexes = ref([])
const fusionSelectedIndexes = ref([])

const currentEditableLines = computed(() => {
  if (isEditing.value) {
    return editedLines.value
  }
  return annotationLines.value
})

const loadDetail = async () => {
  try {
    loading.value = true
    const res = await imagesApi.detail(lineId, currentProject.value)
    const data = res.data?.data
    if (!data) {
      throw new Error('数据为空')
    }
    charCount.value = Number(data.char_count) || 0
    chars.value = Array.isArray(data.annotation?.chars) ? data.annotation.chars : []
    ruleLines.value = Array.isArray(data.rule_lines) ? data.rule_lines : []
    modelLines.value = Array.isArray(data.model_lines) ? data.model_lines : []
    fusionLines.value = Array.isArray(data.fusion_lines) ? data.fusion_lines : []
    annotationLines.value = Array.isArray(data.annotation?.lines) ? data.annotation.lines : []
    is_annotated.value = Boolean(data.is_annotated)
    is_postponed.value = Boolean(data.is_postponed)
    editedLines.value = [...annotationLines.value]
  } catch (error) {
    chars.value = []
    ruleLines.value = []
    modelLines.value = []
    fusionLines.value = []
    annotationLines.value = []
    alert('加载数据失败，请重试')
    console.error('加载数据失败:', error)
  } finally {
    loading.value = false
  }
}

const startEditing = () => {
  isEditing.value = true
  editedLines.value = [...annotationLines.value]
}

const handleLinesChange = (lines) => {
  editedLines.value = lines
}

const handleSelectedIndexesChange = (indexes) => {
  selectedIndexes.value = indexes
}

const copySelectedLines = (scheme) => {
  let sourceLines = []
  let selectedIdxs = []
  
  switch (scheme) {
    case 'rule':
      sourceLines = ruleLines.value
      selectedIdxs = ruleSelectedIndexes.value
      break
    case 'model':
      sourceLines = modelLines.value
      selectedIdxs = modelSelectedIndexes.value
      break
    case 'fusion':
      sourceLines = fusionLines.value
      selectedIdxs = fusionSelectedIndexes.value
      break
  }

  const selectedLines = selectedIdxs.map(idx => sourceLines[idx])
  
  if (selectedLines.length === 0) {
    alert('请先在预览区域选中要复制的切割线')
    return
  }

  isEditing.value = true
  
  const existingPositions = new Set(editedLines.value.map(line => line.pos))
  
  const newLines = selectedLines.filter(line => !existingPositions.has(line.pos))
  
  editedLines.value = [...editedLines.value, ...newLines]
  editedLines.value.sort((a, b) => a.pos - b.pos)
  
  alert(`已复制 ${newLines.length} 条切割线到编辑区域`)
}

const useScheme = (scheme) => {
  let sourceLines = []
  
  switch (scheme) {
    case 'rule':
      sourceLines = ruleLines.value
      break
    case 'model':
      sourceLines = modelLines.value
      break
    case 'fusion':
      sourceLines = fusionLines.value
      break
  }

  isEditing.value = true
  editedLines.value = JSON.parse(JSON.stringify(sourceLines))
  selectedIndexes.value = []
  
  alert(`已使用${scheme === 'rule' ? '规则' : scheme === 'model' ? '模型' : '融合'}切割方案`)
}

const deleteSelected = () => {
  if (selectedIndexes.value.length === 0) {
    alert('请先选中要删除的切割线')
    return
  }

  const newLines = [...editedLines.value]
  selectedIndexes.value.sort((a, b) => b - a).forEach(i => newLines.splice(i, 1))
  editedLines.value = newLines
  selectedIndexes.value = []
}

const clearAll = () => {
  if (!confirm('确定要清空编辑区所有切割线吗？此操作不可撤销。')) {
    return
  }
  editedLines.value = []
  selectedIndexes.value = []
}

const notifyStatusChange = () => {
  const statusData = {
    line_id: lineId,
    project_id: currentProject.value,
    is_annotated: is_annotated.value,
    is_postponed: is_postponed.value,
    timestamp: Date.now()
  }
  localStorage.setItem('annotation_status_change', JSON.stringify(statusData))
}

const saveAnnotation = async () => {
  try {
    const data = {
      lines: editedLines.value
    }
    if (currentProject.value) {
      data.project_id = currentProject.value
    }
    await imagesApi.annotate(lineId, data)
    isEditing.value = false
    is_annotated.value = true
    is_postponed.value = false
    annotationLines.value = [...editedLines.value]
    await loadDetail()
    notifyStatusChange()
    alert('标注保存成功')
  } catch (error) {
    alert('保存失败，请重试')
    console.error('保存失败:', error)
  }
}

const cancelEditing = () => {
  isEditing.value = false
  editedLines.value = [...annotationLines.value]
  selectedIndexes.value = []
}

const handlePostpone = async () => {
  try {
    await imagesApi.postpone(lineId, currentProject.value)
    is_postponed.value = true
    is_annotated.value = false
    notifyStatusChange()
    alert('已标记为暂不标注')
  } catch (error) {
    alert('操作失败')
    console.error('操作失败:', error)
  }
}

const handleUnpostpone = async () => {
  try {
    await imagesApi.unpostpone(lineId, currentProject.value)
    is_postponed.value = false
    notifyStatusChange()
    alert('已取消暂不标注')
  } catch (error) {
    alert('操作失败')
    console.error('操作失败:', error)
  }
}

const goBack = () => {
  router.push({ path: '/segment', query: { project_id: currentProject.value } })
}

const zoomIn = () => {
  zoomLevel.value = Math.min(zoomLevel.value + 0.1, 3.0)
}

const zoomOut = () => {
  zoomLevel.value = Math.max(zoomLevel.value - 0.1, 0.5)
}

const resetZoom = () => {
  zoomLevel.value = 1.0
}

onMounted(() => loadDetail())
</script>

<style scoped>
.label-page { padding: 20px; max-width: 1600px; margin: 0 auto; box-sizing: border-box; }
.header { display: flex; align-items: center; margin-bottom: 30px; padding-bottom: 15px; border-bottom: 1px solid #e8e8e8; gap: 16px; flex-wrap: wrap; }
.header h2 { margin: 0; font-size: 24px; font-weight: 600; color: #333; }
.char-count { font-size: 16px; color: #666; background: #f0f0f0; padding: 4px 12px; border-radius: 12px; }
.loading-container { display: flex; align-items: center; justify-content: center; min-height: 600px; }
.content { display: flex; flex-direction: column; gap: 20px; }

.section-card { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.09); overflow: hidden; }
.section-card.editable-card { border: 2px solid #1890ff; }
.section-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; background: #fafafa; border-bottom: 1px solid #e8e8e8; }
.section-header h3 { margin: 0; font-size: 18px; font-weight: 600; color: #333; }
.line-color-selector { display: flex; align-items: center; gap: 8px; }
.color-label { font-size: 13px; color: #666; }
.zoom-controls { display: flex; align-items: center; gap: 4px; margin-left: 16px; }
.zoom-label { font-size: 13px; color: #666; }
.zoom-value { font-size: 13px; color: #333; min-width: 50px; text-align: center; }
.line-count { font-size: 14px; color: #666; background: #f0f0f0; padding: 4px 12px; border-radius: 12px; margin-right: 8px; }
.section-body { padding: 20px; overflow-x: auto; max-height: 800px; }

.header-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

.original-image { max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; }

.edit-hint { margin-bottom: 12px; padding: 10px 15px; background: #f6ffed; border: 1px solid #b7eb8f; border-radius: 4px; font-size: 14px; color: #52c41a; }
.edit-hint strong { color: #1890ff; }

.editable-canvas-wrapper { overflow: auto; max-height: 600px; }
.editable-canvas-wrapper :deep(canvas) { max-width: none; height: auto !important; }

.action-bar { display: flex; flex-wrap: wrap; gap: 24px; padding: 20px; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.09); }
.action-section { display: flex; align-items: center; gap: 12px; }
.section-label { font-size: 14px; font-weight: 600; color: #666; padding-right: 8px; border-right: 1px solid #e8e8e8; }

:deep(canvas) { max-width: 100%; height: auto !important; }
</style>
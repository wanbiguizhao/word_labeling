import axios from 'axios'

const API_BASE_URL = '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000
})

api.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

export const projectsApi = {
  list() {
    return api.get('/projects')
  },
  create(data) {
    return api.post('/projects', data)
  },
  detail(projectId) {
    return api.get(`/projects/${projectId}`)
  },
  delete(projectId) {
    return api.delete(`/projects/${projectId}`)
  },
  switch(projectId) {
    return api.post(`/projects/${projectId}/switch`)
  }
}

export const selectorsApi = {
  strategies() {
    return api.get('/selectors/strategies')
  },
  preview(data) {
    return api.post('/selectors/preview', data)
  }
}

export const imagesApi = {
  list(params) {
    return api.post('/images', params)
  },
  detail(lineId, projectId) {
    const params = projectId ? { project_id: projectId } : {}
    return api.get(`/images/${lineId}/detail`, { params })
  },
  raw(lineId, projectId) {
    const params = new URLSearchParams()
    if (projectId) {
      params.set('project_id', projectId)
    }
    return `/api/images/${lineId}/raw?${params.toString()}`
  },
  annotate(lineId, data) {
    return api.post(`/images/${lineId}/annotate`, data)
  },
  postpone(lineId, projectId) {
    const params = projectId ? { project_id: projectId } : {}
    return api.post(`/images/${lineId}/postpone`, {}, { params })
  },
  unpostpone(lineId, projectId) {
    const params = projectId ? { project_id: projectId } : {}
    return api.post(`/images/${lineId}/unpostpone`, {}, { params })
  }
}

export const annotationApi = {
  stats(projectId) {
    const params = projectId ? { project_id: projectId } : {}
    return api.get('/annotation/stats', { params })
  },
  registryStats() {
    return api.get('/annotation/registry-stats')
  },
  priorityQueue(limit = 20, projectId) {
    const params = { limit }
    if (projectId) {
      params.project_id = projectId
    }
    return api.get('/annotation/priority-queue', { params })
  },
  next(projectId) {
    const params = projectId ? { project_id: projectId } : {}
    return api.get('/annotation/next', { params })
  }
}
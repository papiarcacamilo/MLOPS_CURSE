// Pipeline de aprobacion del modelo de riesgo crediticio.
//
// GitHub Actions protege el pull request antes del merge. Este pipeline actua
// despues: revisa lo que llego a develop o master y, en master, pide la aprobacion
// de una persona. Cualquier paso que falle rechaza la corrida.

pipeline {
    agent any

    options {
        // El clon es un paso visible de la corrida, no algo implicito.
        skipDefaultCheckout(true)
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    triggers {
        // Jenkins corre en localhost y GitHub no puede llamarlo con un webhook:
        // es Jenkins quien pregunta si hay commits nuevos.
        pollSCM('H/5 * * * *')
    }

    stages {
        stage('Clonar repositorio') {
            steps {
                deleteDir()
                script {
                    def repo = checkout scm
                    env.RAMA = repo.GIT_BRANCH
                    env.COMMIT = repo.GIT_COMMIT
                }
                echo "Rama ${env.RAMA}, commit ${env.COMMIT}"
            }
        }

        stage('Validaciones') {
            agent {
                docker {
                    image 'python:3.11-slim'
                    // reuseNode conserva el workspace del clon. El volumen evita
                    // descargar todas las dependencias en cada corrida.
                    reuseNode true
                    args '-v mlops_pip_cache:/root/.cache/pip'
                }
            }
            environment {
                PIP_CACHE_DIR = '/root/.cache/pip'
                PIP_ROOT_USER_ACTION = 'ignore'
                PIP_DISABLE_PIP_VERSION_CHECK = '1'
            }
            options {
                timeout(time: 20, unit: 'MINUTES')
            }
            stages {
                stage('Validar estructura') {
                    // Va antes de instalar: si la estructura esta rota, rechaza en segundos.
                    steps {
                        sh 'python ci/validar_estructura.py'
                    }
                }
                stage('Preparar entorno') {
                    steps {
                        sh 'pip install --quiet -r requirements.txt'
                    }
                }
                stage('Correr pruebas') {
                    steps {
                        sh 'python -m pytest -q --junitxml=reporte_pruebas.xml'
                    }
                    post {
                        always {
                            junit testResults: 'reporte_pruebas.xml', allowEmptyResults: true
                        }
                    }
                }
                stage('Validar metricas') {
                    steps {
                        sh 'python ci/validar_metricas.py'
                    }
                }
            }
        }

        stage('Aprobacion humana') {
            // Solo master llega a produccion. Queda fuera del contenedor para no
            // mantenerlo encendido mientras se espera a la persona.
            when {
                expression { env.RAMA == 'origin/master' }
            }
            options {
                timeout(time: 1, unit: 'HOURS')
            }
            steps {
                input message: "Aprobar el commit ${env.COMMIT} para produccion?",
                      ok: 'Aprobar',
                      submitter: 'camilo'
            }
        }
    }

    post {
        success {
            echo "APROBADO: ${env.RAMA}, commit ${env.COMMIT}"
        }
        failure {
            echo "RECHAZADO: ${env.RAMA}, commit ${env.COMMIT}. Revisar la etapa que fallo."
        }
        aborted {
            echo "SIN APROBAR: ${env.RAMA}, commit ${env.COMMIT}. Se cancelo o vencio la espera."
        }
    }
}

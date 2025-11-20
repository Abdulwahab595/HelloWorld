pipeline {
    agent any

    environment {
        PROJECT = "Static-3"
        ORG = "mightykarim"
    }

    stages {

        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/mightykarim/Static-3.git'
            }
        }

        stage('Build') {
            steps {
                echo "Build step..."
            }
        }
        stage('SonarCloud Analysis') {
            steps {
                withCredentials([string(credentialsId: 'sonarcloud-token', variable: 'TOKEN')]) {
                    withSonarQubeEnv('SonarCloud') {
                        bat """
                            ${tool 'SonarScanner'}\\bin\\sonar-scanner.bat ^
                              -Dsonar.projectKey=Static-3 ^
                              -Dsonar.organization=mightykarim ^
                              -Dsonar.sources=. ^
                              -Dsonar.host.url=https://sonarcloud.io ^
                              -Dsonar.login=%TOKEN% ^
                              -Dsonar.c.file.suffixes=- ^
                              -Dsonar.cpp.file.suffixes=- ^
                              -Dsonar.objc.file.suffixes=-
                        """
                    }
                }
            }
        }


        stage('Quality Gate') {
            steps {
                timeout(time: 2, unit: 'MINUTES') {
                    script {
                        def qg = waitForQualityGate()
                        if (qg.status == 'NONE') {
                            echo "Warning: No quality gate configured. Proceeding with build."
                        } else if (qg.status != 'OK') {
                            error "Pipeline aborted due to quality gate failure: ${qg.status}"
                        }
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                echo "Deployment step..."
            }
        }
    }
}

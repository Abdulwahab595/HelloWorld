pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                echo 'Building..'
                // Here you can define commands for your build
            }
        }

        stage('Test') {
            steps {
                echo 'Testing..'
                // Here you can define commands for your tests
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploying....'
                // Here you can define commands for your deployment
            }
        }
    }

    // Post-build actions
    post {
        always {
            echo 'Post build condition running'
        }
        success {
            echo 'This runs only if the build succeeded'
        }
        failure {
            echo 'Post action if build failed'
        }
        unstable {
            echo 'This runs if the build is unstable'
        }
    }
}

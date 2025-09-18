git reset --hard origin/main
git pull origin main
git reset --hard origin/main
git pull origin main
gcloud config set project ai-my-analyst
gcloud alpha resource-manager liens create     --project=ai-my-analyst-b01fd     --restrictions=resourcemanager.projects.delete     --reason="This project is protected by a lien"     --origin=david@buadze.com
$ gcloud config set account david@buadze.com
gcloud auth login
gcloud alpha resource-manager liens create     --project=ai-my-analyst-b01fd     --restrictions=resourcemanager.projects.delete     --reason="This project is protected by a lien"     --origin=david@buadze.com
gcloud alpha resource-manager liens list
cd ~
tar -czvf ailbee-project.tar.gz frontend backend
tar -czvf ailbee-project.tar.gz frontend backend functions
tar -czvf ailbee-project.tar.gz frontend backend functions
tar -czvf david-project.tar.gz david/
tar -xzvf david-project.tar.gz
zip -r david-project.zip david/
unzip david-project.zip
tar -czvf david-project.tar.gz --exclude="david/node_modules" --exclude="david/.git" david/
zip -r david-project.zip david/ -x "david/node_modules/*" -x "david/.git/*"
tar -czvf david-project.tar.gz david/
git add .  
git commit -m “Update”  
git push origin main
git add .  
git commit -m “Update”  
git push origin main
gcloud auth login
gcloud config set project ailbee
git add .  
git commit -m “Update”  
git push origin main
git add .  
git commit -m “Update”  
git push origin main
git config --global user.email "david@buadze.com"
git config --global user.name "davidbuadze"
git add .  
git commit -m “Update”  
git push origin main

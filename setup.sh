# mkdir -p ~/.streamlit/                                               
# echo "\                       
# [server]\n\                       
# port = $PORT\n\                       
# enableCORS = false\n\                       
# headless = true\n\                       
# \n\                       
# " > ~/.streamlit/config.toml

mkdir -p ~/.streamlit/
 echo “
 [server]
 headless = true
 enableCORS=false
 port = $PORT
#  “ > ~/.streamlit/config.toml
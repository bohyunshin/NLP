## 삽질 기록

### 4.27: glove_python 설치하기 징하게도 안된다.
첫 번째 에러는 ERROR: Command errored out with exit status 1:라고 뜨면서 

usr/local/Cellar/gcc/9.3.0_1/bin/g++-9 

얘를 찾을 수 없다고 떠서 애를 먹었는데 

export CC=/usr/local/Cellar/gcc/9.3.0_1/bin/g++-9 

얘를 적어주면 된다. 우선 

brew info gcc 

에서 gcc의 버젼을 확인해주고, 9.3.0이라면 g++-9를, 8.3.0이라면 g++-8을 적어주면 됨. 
여기서 9.3.0인지 9.3.0_1인지는 brew info gcc를 자세히 살펴보면 있음.

이거 해결했는데 다음 오류는?!

glove/glove_cython.c: In function 'void __Pyx_ExceptionSave(PyObject**, PyObject**, PyObject**)':

이 오류보고 cython과 python 연동에 문제가 있다고 판단함. 그래서 cython에 관한 것을 찾아봤는데 

https://github.com/cython/cython/issues/1978

이런 이슈가 있었음. 즉, 파이썬 3.7에서는 cython이 호환 안 된다는 것임. 그래서 아예 가상환경 만들고, 여기에 파이썬 3.6으로해서 해봤더니 드디어 설치가 됐다!!!!!!!

### 4.27: 가상환경에서 jupyter notebook?
위에서 가상환경을 만들었으니 이 가상환경에서 쥬피터를 키고 싶었다. 그래서 jupter notebook을 쳤더니 jupyter 명령어를 인식하지 못하네? 그런데 conda 는 인식을 함! 그래서 구글링을 해보니 가상환경에서 쥬피터를 킬 때에는 jupyter 라이브러리도 다시 깔아줘야한다고 함. 그래서 pip install jupyter해주고 쥬피터 켜봤더니 잘 된다~

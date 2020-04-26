## 삽질 기록

### 4.27: glove_python 설치하기 징하게도 안된다.
첫 번째 에러는 ERROR: Command errored out with exit status 1:라고 뜨면서 <br>

usr/local/Cellar/gcc/9.3.0_1/bin/g++-9 <br>

얘를 찾을 수 없다고 떠서 애를 먹었는데 <br>

export CC=/usr/local/Cellar/gcc/9.3.0_1/bin/g++-9 <br>

얘를 적어주면 된다. 우선 <br>

brew info gcc <br>

에서 gcc의 버젼을 확인해주고, 9.3.0이라면 g++-9를, 8.3.0이라면 g++-8을 적어주면 됨. <br>
여기서 9.3.0인지 9.3.0_1인지는 brew info gcc를 자세히 살펴보면 있음.

이거 해결했는데 다음 오류는?!

glove/glove_cython.c: In function 'void __Pyx_ExceptionSave(PyObject**, PyObject**, PyObject**)':

이 오류보고 cython과 python 연동에 문제가 있다고 판단함. 그래서 cython에 관한 것을 찾아봤는데 

https://github.com/cython/cython/issues/1978

이런 이슈가 있었음. 즉, 파이썬 3.7에서는 cython이 호환 안 된다는 것임. 그래서 아예 가상환경 만들고, 여기에 파이썬 3.6으로해서 해봤더니 드디어 설치가 됐다!!!!!!!

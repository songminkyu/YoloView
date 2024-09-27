# 이 파일은 전역 변수에 대한 파일 간 액세스를 구현하는 데 사용됩니다.
def _init   ():  # 초기화
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    #   전역 변수 정의
    _global_dict[key] = value


def get_value(key):
    #   전역 변수를 얻습니다. 존재하지 않는 경우 해당 변수를 읽을 수 없다는 메시지가 표시됩니다.
    try:
        return _global_dict[key]
    except:
        return None
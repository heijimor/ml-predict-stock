import yfinance as yf
import pandas as pd
from prometheus_client import start_http_server, Summary, Counter, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

class StockManager:
  
  # model = load_model("saved_model/model_transfer_learning.pth")

  # REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
  # INFERENCE_COUNT = Counter('inference_count', 'Total number of inferences made')
  # MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy over time')
  
  def __init__(self):
    pass
  
  def collect(
    self,
    ticker: str,
    start_date: str,
    end_date: str
  ) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date)
    print(f'Yahoo Downloaded: \n ${df}')
    data = df[['Close']]
    print(f'Yahoo Close: \n ${data}')
    return data
  
  def load_model(self, model):
    # model = models.resnet18(pretrained=False)
    # model.fc = nn.Linear(model.fc.in_features, 2)
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # return model
    pass
  
  def predict(self, data):
    # from prometheus_client import start_http_server, Summary, Counter, Gauge, generate_latest
    # from prometheus_client import CONTENT_TYPE_LATEST

    # from .model import load_model, predict

    # model = load_model("saved_model/model_transfer_learning.pth")

    # REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
    # INFERENCE_COUNT = Counter('inference_count', 'Total number of inferences made')
    # MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy over time')

    # # Função para inferência com monitoramento
    # @REQUEST_TIME.time()
    # def process_request(image):
    #     INFERENCE_COUNT.inc()
    #     prediction = predict(model, image)
    #     # Exemplo de métrica para monitoramento de acurácia (valor fixo para o exemplo)
    #     accuracy = 0.9  # Supondo que esta é uma métrica fixa ou calculada de alguma forma
    #     MODEL_ACCURACY.set(accuracy)
    #     return prediction
    pass
  
  def normalize(self):
    pass
  
  def sequencialize(self):
    pass
  
  def prepare(self):
    pass
  
  def build(self):
    pass
  
  def train(self):
    pass
  
  def test(self):
    pass
  
  def save(self):
    pass
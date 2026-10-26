# ReCoRDS (Streamlit + RocketCEA) — Docker で公開する場合 (Render / Hugging Face Spaces など) 用
# RocketCEA は Linux 向けのビルド済みパッケージが無いため、gfortran でソースからビルドする
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends gfortran gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Spaces はユーザー ID 1000 で実行される。RocketCEA は HOME 以下にデータを書くので書き込み可能にしておく
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    && python -c "from rocketcea.cea_obj import CEA_Obj; print('RocketCEA OK:', CEA_Obj(oxName='LOX', fuelName='CH4').get_Isp(1000, 3, 40))"

COPY --chown=user app.py .

# ホスティング側が PORT 環境変数でポートを指定する場合 (Render など) はそれに従う
EXPOSE 8501
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]

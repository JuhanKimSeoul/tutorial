# Problem-Solving

## 1. 여러거래소의 kline history 데이터를 비교

### Problem 

- 1m, 3m, 15m.. 분 단위와 1h, 1d, 1w 데이터는 거래소간 오차가 나지 않는다.
- 하지만, 자주 보는 interval 중 4h는 거래소마다 기준시간이 달라서 데이터 전처리가 필요하다.
- 업비트와 코인원은 UTC 00:00시를 기준으로 한다.
- 빗썸과 코빗은 UTC KST 00:00시를 기준으로 한다.
- 예시1) 현재 UTC KST가 20시라면, 업비트에서 4h의 마지막 캔들은 17:00이고, 빗썸의 마지막 캔들은 20:00이다.
- 예시2) 현재 UTC KST가 21시라면, 업비트에서 4h의 마지막 캔들은 21:00이고, 빗썸의 마지막 캔들은 20:00이다.

### Solution

해외 거래소는 UTC 00:00시를 기준으로 한다. 따라서, 거래소 간 시간 sync를 맞추려면, 빗썸과 코빗의 기준을 UTC KST -> UTC로 바꿔줘야 한다.
단순하게 가져온 데이터를 통해 pd.Timedelta를 하면 안된다. 위의 예시1,2에서 보듯이, 최신데이터가 시간에 따라서 있을 수도 있고, 없을 수도 있기 때문이다.
결국 UTC time을 기준으로 해야하기 때문에 UTC time을 구해서 데이터 sync를 맞춰야 한다. 
예시 1)에서 UTC KST 20:00이면 UTC 11:00이므로, UTC 00시를 기준으로 4h씩 회전한다고 치면, 아직 2회전한 것이므로, UTC 08:00시가 기준이다. 
예시 2)에서 UTC KST 21:00이면 UTC 12:00이므로, UTC 00시를 기준으로 4h씩 회전한다고 치면, 3회전한 것이므로, UTC 12:00시가 기준이다.
따라서, kline history데이터에서 timestamp항목으로 UTC시간을 구한다음, 4로 나눈 몫을 통해 회전수를 구하여, UTC기준의 4h기준시를 구해서 pd.Timedelta를 해주자.

UTC         upbit       bithumb
T   11:00               T 20:00
T   07:00               T 16:00
T   03:00               T 12:00
T-1 23:00               T 08:00
T-1 19:00               T 04:00
T-1 15:00               T 00:00

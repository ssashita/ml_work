# NSE Stock Advise Analystics

## The data is completely fictitious and anonymized - Stock Advisor names are numbers and there are no dates or prices. Only information about whether an advise hit the target or stoploss


|index| buysell |durationtype| 	advisor| 	otheradvices| 	symbolname| 	success| 	niftysentiment|
|-----|---------|------------|----------|----------------|------------|-----------|------------------|
|0 	   |     1 |	    2 |            169 |	     False |	     ADANIENT| 	 False |	        1     |
|1 	   |     1 |     3 |	           161 |	     True 	|      GODREJCP| 	 True 	|         2     |
|2 	   |     1 |	    2 |	           39 	|      False |	     CIPLA 	 |   False |	        2     |
   etc
   
#### What the attributes (columns) mean

  *buysell* - It's a buy advise if 1 , sell advise if 2
  
  *durationtype* - It's a long term advise if 3, short term if 2 and day trade advise if 1
  
  *advisor* - The id of the advisor who gave the advise to buy or sell the stock
  
  *otheradvices* - If True it means there were other advises within 7 days of this one for the same stock symbol
  
  *symbolname* - Name of the stock symbol in NSE
  
  *success* - If True, it means the stock value hit the Target price mentioned in the advise, if False, it means that Stoploss was hit
  
  *niftysentiment* - 1 means that on the day that the stock hit either Stoploss or Target price, Nifty 50 index had opened higher than the previous trading day's close. 2 means otherwise
  
  
##### The 'success' column makes for a good predicted (label) variable. Typically, if some advisor gives a new advise regarding some stock we would like to know if it will be a success or failure.






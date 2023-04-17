library("RSQLite")
library("sqldf")
library("tcltk")
library("data.table")
library("caret")
library("randomForest")
#connect to the database
db <- dbConnect(SQLite(), dbname="../input/database.sqlite")
#List of tables
dbListTables(db)

#Load data
trainSearchStream<-dbGetQuery(db, "SELECT trainSearchStream.SearchID,trainSearchStream.AdID,trainSearchStream.Position,trainSearchStream.objectType, trainSearchStream.HistCTR,trainSearchStream.IsClick,SearchInfo.SearchDate, SearchInfo.UserID,SearchInfo.CategoryID as SearchCategoryID,AdsInfo.Price,AdsInfo.LocationID,AdsInfo.CategoryID FROM trainSearchStream, SearchInfo, UserInfo, AdsInfo where trainSearchStream.SearchID = SearchInfo.SearchID and SearchInfo.UserID=UserInfo.UserID and trainSearchStream.AdID=AdsInfo.AdID and AdsInfo.IsContext=1 limit 1000000")

#Number of PhoneRequest per user
NumPhoneRequest<-dbGetQuery(db,"select UserID, count(UserID) as NumPhoReq from PhoneRequestsStream group by UserID ")

#Number of View per user
NumViews<-dbGetQuery(db,"select UserID, count(UserID) as NumView from VisitsStream group by UserID ")

#length(NumPhoneRequest)
#head(NumPhoneRequest)

trainSearchStream[is.na(trainSearchStream)]<--1
#Convert Position and Price to numeric (train)
position <- as.factor(trainSearchStream$Position)
price <- as.numeric(trainSearchStream$Price)
isClick<-as.numeric(trainSearchStream$IsClick)
HistCTR<-as.numeric(trainSearchStream$HistCTR)
LocationID<-as.factor(trainSearchStream$LocationID)
CategoryID<-as.factor(trainSearchStream$CategoryID)
SearchCategoryID<-as.factor(trainSearchStream$SearchCategoryID)


#DataFrame for train data
#data_train<-data.frame("isClick"=isClick,"Position"=position,"Price"=price, "HistCTR"=HistCTR, "LocationID"=LocationID,"CategoryID"=CategoryID, "SearchCategoryID"=SearchCategoryID)
data_train<-data.frame("isClick"=isClick,"Position"=position,"Price"=price, "HistCTR"=HistCTR,"CategoryID"=CategoryID, "SearchCategoryID"=SearchCategoryID,"UserID"=trainSearchStream$UserID)
#data_train<-sqldf("select data_train.UserID, isClick, Position, price, HistCTR,LocationID, CategoryID, SearchCategoryID, NumPhoReq from data_train left join NumPhoneRequest on data_train.UserID=NumPhoneRequest.UserID")
#data_train<-sqldf("select isClick, Position, price, HistCTR,LocationID, CategoryID, SearchCategoryID, NumPhoReq,NumView from data_train left join NumViews on data_train.UserID=NumViews.UserID")
thousand <- 1000
million  <- thousand * thousand 
billion  <- thousand * million

data_train$UserID<-NULL



m<-nrow(data_train)

sampleSize <- 0.5* million
sampleRatio <- sampleSize / m
sampleIndex <- createDataPartition(data_train$isClick, p = sampleRatio, list=FALSE)
data_train <- data_train[as.vector(sampleIndex), ]
data_train[is.na(data_train)]<-(-1)
head(data_train)


#model
model <- randomForest(isClick ~., data = data_train, ntree=50,
                        do.trace=2,replace=FALSE,verboseiter=FALSE)
#model<-glm(formula = isClick ~ position + HistCTR + SearchCategoryID +NumView, data=data_train, family = binomial("logit"))
summary(model)
rm(data_train)
#Convert Position and Price to numeric (test)
testSearchStream<-dbGetQuery(db, "SELECT testSearchStream.TestId, testSearchStream.Position, testSearchStream.HistCTR,AdsInfo.Price,AdsInfo.LocationID,AdsInfo.CategoryID, SearchInfo.UserID, SearchInfo.CategoryID as SearchCategoryID FROM testSearchStream, SearchInfo, AdsInfo where testSearchStream.SearchID = SearchInfo.SearchID and testSearchStream.AdID=AdsInfo.AdID and AdsInfo.IsContext=1 ")
position <- as.factor(testSearchStream$Position)
price <- as.numeric(testSearchStream$Price)
HistCTR<-as.numeric(testSearchStream$HistCTR)
LocationID<-as.factor(testSearchStream$LocationID)
CategoryID<-as.factor(testSearchStream$CategoryID)
SearchCategoryID<-as.factor(testSearchStream$SearchCategoryID)

length(LocationID)
length(CategoryID)

#DataFrame for test data
data_test<-data.frame("Position"=position,"Price"=price, "HistCTR"=HistCTR,"CategoryID"=CategoryID, "SearchCategoryID"=SearchCategoryID,"UserID"=testSearchStream$UserID)
#data_test<-sqldf("select data_test.UserID,Position, price, HistCTR,LocationID, CategoryID, SearchCategoryID, NumPhoReq from data_test left join NumPhoneRequest on data_test.UserID=NumPhoneRequest.UserID")
#data_test<-sqldf("select Position, price, HistCTR,LocationID, CategoryID, SearchCategoryID, NumPhoReq,NumView from data_test left join NumViews on data_test.UserID=NumViews.UserID")
data_test[is.na(data_test)]<-(-1)
head(data_test)
#prediction
predictions<-predict(model,data_test,type="response")
predictions[is.na(predictions)] <- 0


#submission
write.csv(data.frame("ID"=testSearchStream$TestId, "IsClick"=predictions),"prediction.csv",quote=F,row.names=F);
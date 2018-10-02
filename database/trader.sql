create database if not exists trader;
use trader;

-- TODO expand this to futures and ETF
create table security(
    symbol varchar(5) not null,
    name varchar(60),
--    exchange varchar(6) not null,
    sector varchar(30),
    industry varchar(30),
    primary key(symbol)
);

-- Related securities that show correlated price patterns
create table trendswith(
    leader varchar(5) not null,
    follower varchar(5) not null,
    primary key(leader, follower)
);

create table buy(
    positionId int not null,
    symbol varchar(5) not null,
    avgPrice decimal(8,4) not null,
    shares int not null,
    primary key(positionId)
);

create table sell(
    positionId int not null,
    symbol varchar(5) not null,
    avgPrice decimal(8,4) not null,
    shares int not null,
    primary key(positionId)
);

create table history(
    symbol varchar(5) not null,
    startTime datetime not null,
    endTime datetime not null,
    low decimal(8,4),
    high decimal(8,4),
    open decimal(8,4) not null,
    close decimal(8,4) not null,
    vwap decimal(8,4),
    volume int not null,
    primary key(symbol, startTime, endTime)
);

create table daily(
    symbol varchar(5) not null,
    primary key(symbol)
);

create table intraday(
    symbol varchar(5) not null,
    primary key(symbol)
);

alter table trendswith add constraint twl
foreign key(leader) references security(symbol) on delete cascade;

alter table trendswith add constraint twf
foreign key(follower) references security(symbol) on delete cascade;

alter table buy add constraint buys
foreign key(symbol) references history(symbol) on delete cascade;

alter table sell add constraint sells
foreign key(symbol) references history(symbol) on delete cascade;

alter table daily add constraint dailyfk
foreign key(symbol) references history(symbol) on delete cascade;

alter table intraday add constraint intradayfk
foreign key(symbol) references history(symbol) on delete cascade;